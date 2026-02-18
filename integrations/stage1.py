"""
Stage 1 Integration: Prompt Analysis with SCE and Φ₁.₅

Integrates RESE's Symbolic Constraint Engine (SCE) and Tacit Assumption Miner (Φ₁.₅)
with E2E Stage 1 Prompt Analysis.

Architecture:
┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐
│  Prompt Input    │───▶│   SCE (Φ₁)       │───▶│  Φ₁.₅ Feedback   │
│  Analysis        │    │  Constraint      │    │  Loop            │
└──────────────────┘    └──────────────────┘    └──────────────────┘
                               │
                               ▼
                        ┌──────────────────┐
                        │  Refined         │
                        │  Constraints     │
                        └──────────────────┘

Author: Agent A4 (Stage Integration Lead)
Created: 2025-12-31
Status: 🟢 Active Implementation
Target: 1.5 hours implementation
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Tuple
from enum import Enum
from datetime import datetime
import json
from pathlib import Path

# Import RESE components
import sys
from pathlib import Path

# Try to import RESE components, use stubs if not available
try:
    sys.path.insert(0, str(Path(__file__).parent.parent / "rese"))
    from core.symbolic_constraint_engine import (
        SymbolicConstraintEngine, Constraint, ConstraintType
    )
    SCE_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    SCE_AVAILABLE = False
    # Create stub classes for graceful degradation
    class ConstraintType:
        REQUIRED = "required"
        OPTIONAL = "optional"
        PREFERRED = "preferred"

    class Constraint:
        def __init__(self, *args, **kwargs):
            self.type = kwargs.get('type', 'optional')
            self.description = kwargs.get('description', '')

    class SymbolicConstraintEngine:
        def __init__(self, *args, **kwargs):
            self.available = False

try:
    from phase1.tacit_assumption_miner import (
        Phi15Engine, AssumptionType, ErrorType, NullResult
    )
    PHI15_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    PHI15_AVAILABLE = False
    # Create stub classes
    class AssumptionType:
        IMPLICIT = "implicit"
        EXPLICIT = "explicit"

    class ErrorType:
        OMISSION = "omission"
        COMMISSION = "commission"

    class NullResult:
        pass

    class Phi15Engine:
        def __init__(self, *args, **kwargs):
            self.available = False

try:
    from phase1.cognitive_biases import CognitiveBiasDetector
    BIAS_DETECTOR_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    BIAS_DETECTOR_AVAILABLE = False
    class CognitiveBiasDetector:
        def __init__(self, *args, **kwargs):
            self.available = False


# ============================================================================
# Enums and Data Structures
# ============================================================================

class PromptAnalysisStatus(Enum):
    """Status of prompt analysis"""
    ANALYZING = "analyzing"
    CONSTRAINTS_EXTRACTED = "constraints_extracted"
    ASSUMPTIONS_MINED = "assumptions_mined"
    BIASES_DETECTED = "biases_detected"
    REFINING = "refining"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class PromptInput:
    """Input prompt from user"""
    text: str
    domain: str = "general"
    context: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ConstraintExtraction:
    """Extracted constraint from prompt"""
    id: str
    text: str
    type: ConstraintType
    confidence: float
    source_span: Tuple[int, int]  # (start, end) in prompt text
    dependencies: List[str] = field(default_factory=list)


@dataclass
class PromptAnalysisResult:
    """Result from Stage 1 prompt analysis"""
    status: PromptAnalysisStatus
    constraints: List[Constraint]
    assumptions: List[Dict[str, Any]]
    bias_report: Optional[Dict[str, Any]] = None
    refined_prompt: Optional[str] = None
    confidence_score: float = 0.0
    analysis_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'status': self.status.value,
            'constraints': [
                {
                    'id': c.id,
                    'type': c.type.value,
                    'description': c.description,
                    'source': c.source
                }
                for c in self.constraints
            ],
            'assumptions': self.assumptions,
            'bias_report': self.bias_report,
            'refined_prompt': self.refined_prompt,
            'confidence_score': self.confidence_score,
            'analysis_time': self.analysis_time,
            'metadata': self.metadata,
            'errors': self.errors
        }


# ============================================================================
# Main Integration Class
# ============================================================================

class Stage1Integration:
    """
    Stage 1 Integration: Prompt Analysis with RESE components.

    This module integrates:
    1. SCE (Symbolic Constraint Engine) for constraint extraction and management
    2. Φ₁.₅ (Tacit Assumption Miner) for hidden assumption detection
    3. Φ₂ (Cognitive Bias Detector) for bias identification
    4. Feedback loop for prompt refinement

    Workflow:
    1. Extract explicit constraints from prompt
    2. Mine tacit assumptions using Φ₁.₅
    3. Detect cognitive biases using Φ₂
    4. Refine constraints based on findings
    5. Generate refined prompt if needed
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        enable_sce: bool = True,
        enable_phi15: bool = True,
        enable_phi2: bool = True,
        feedback_iterations: int = 2
    ):
        """
        Initialize Stage 1 Integration.

        Args:
            config: Optional configuration dictionary
            enable_sce: Enable Symbolic Constraint Engine
            enable_phi15: Enable Tacit Assumption Miner
            enable_phi2: Enable Cognitive Bias Detector
            feedback_iterations: Number of feedback loop iterations
        """
        self.config = config or {}
        self.enable_sce = enable_sce
        self.enable_phi15 = enable_phi15
        self.enable_phi2 = enable_phi2
        self.feedback_iterations = feedback_iterations

        # Initialize components
        if self.enable_sce:
            self.sce = SymbolicConstraintEngine()

        if self.enable_phi15:
            self.phi15 = Phi15Engine()

        if self.enable_phi2:
            self.phi2 = CognitiveBiasDetector()

        # Analysis history
        self.analysis_history: List[PromptAnalysisResult] = []

    def analyze_prompt(
        self,
        prompt_input: PromptInput,
        use_feedback_loop: bool = True
    ) -> PromptAnalysisResult:
        """
        Analyze prompt and extract constraints.

        Args:
            prompt_input: Input prompt
            use_feedback_loop: Whether to use Φ₁.₅ feedback loop

        Returns:
            PromptAnalysisResult with extracted constraints and insights
        """
        start_time = datetime.now()
        result = PromptAnalysisResult(
            status=PromptAnalysisStatus.ANALYZING,
            constraints=[],
            assumptions=[]
        )

        try:
            # Step 1: Extract explicit constraints using SCE
            if self.enable_sce:
                constraint_extractions = self._extract_constraints(prompt_input)
                result.constraints = self._add_constraints_to_sce(constraint_extractions)
                result.status = PromptAnalysisStatus.CONSTRAINTS_EXTRACTED

            # Step 2: Mine tacit assumptions using Φ₁.₅
            if self.enable_phi15:
                result.assumptions = self._mine_assumptions(
                    prompt_input,
                    result.constraints
                )
                result.status = PromptAnalysisStatus.ASSUMPTIONS_MINED

            # Step 3: Detect cognitive biases using Φ₂
            if self.enable_phi2:
                result.bias_report = self._detect_biases(
                    prompt_input,
                    result.constraints
                )
                result.status = PromptAnalysisStatus.BIASES_DETECTED

            # Step 4: Feedback loop for refinement
            if use_feedback_loop and self.feedback_iterations > 0:
                result = self._feedback_loop_refinement(
                    prompt_input,
                    result,
                    iterations=self.feedback_iterations
                )

            # Calculate confidence score
            result.confidence_score = self._calculate_confidence(result)

            result.status = PromptAnalysisStatus.COMPLETED

        except Exception as e:
            result.status = PromptAnalysisStatus.FAILED
            result.errors.append(str(e))

        # Record time
        end_time = datetime.now()
        result.analysis_time = (end_time - start_time).total_seconds()

        # Store in history
        self.analysis_history.append(result)

        return result

    def _extract_constraints(
        self,
        prompt_input: PromptInput
    ) -> List[ConstraintExtraction]:
        """
        Extract explicit constraints from prompt text.

        Args:
            prompt_input: Input prompt

        Returns:
            List of extracted constraints
        """
        extractions = []

        # Simple constraint extraction patterns
        # In production, this would use NLP/LLM-based extraction

        text = prompt_input.text.lower()

        # Pattern: "must" indicates hard constraints
        if " must " in text or " require " in text or " necessary " in text:
            sentences = text.split('.')
            for i, sentence in enumerate(sentences):
                if " must " in sentence or " require " in sentence:
                    extractions.append(
                        ConstraintExtraction(
                            id=f"hard_constraint_{i}",
                            text=sentence.strip(),
                            type=ConstraintType.HARD,
                            confidence=0.9,
                            source_span=(0, len(sentence))
                        )
                    )

        # Pattern: "should" indicates soft constraints
        if " should " in text or " prefer " in text or " desirable " in text:
            sentences = text.split('.')
            for i, sentence in enumerate(sentences):
                if " should " in sentence or " prefer " in sentence:
                    extractions.append(
                        ConstraintExtraction(
                            id=f"soft_constraint_{i}",
                            text=sentence.strip(),
                            type=ConstraintType.SOFT,
                            confidence=0.8,
                            source_span=(0, len(sentence))
                        )
                    )

        # Pattern: "ideally" indicates preferences
        if " ideally " in text or " optionally " in text or " nice to have " in text:
            sentences = text.split('.')
            for i, sentence in enumerate(sentences):
                if " ideally " in sentence or " optionally " in sentence:
                    extractions.append(
                        ConstraintExtraction(
                            id=f"preference_{i}",
                            text=sentence.strip(),
                            type=ConstraintType.PREFERENCE,
                            confidence=0.7,
                            source_span=(0, len(sentence))
                        )
                    )

        return extractions

    def _add_constraints_to_sce(
        self,
        extractions: List[ConstraintExtraction]
    ) -> List[Constraint]:
        """
        Add extracted constraints to SCE.

        Args:
            extractions: List of constraint extractions

        Returns:
            List of Constraint objects
        """
        constraints = []

        for extraction in extractions:
            constraint = Constraint(
                id=extraction.id,
                type=extraction.type,
                description=extraction.text,
                formalization=self._formalize_constraint(extraction),
                source="prompt_extraction"
            )

            try:
                self.sce.add_constraint(constraint)
                constraints.append(constraint)
            except Exception as e:
                # Log but continue
                logger.error(f"Failed to add constraint {extraction.id}: {e}")

        return constraints

    def _formalize_constraint(
        self,
        extraction: ConstraintExtraction
    ) -> str:
        """
        Convert natural language constraint to formal representation.

        Args:
            extraction: Constraint extraction

        Returns:
            Formal constraint string
        """
        # Simplified formalization
        # In production, this would use more sophisticated formalization
        text = extraction.text.lower()

        if "must" in text:
            return f"∀x, {text.replace('must', '->')}"
        elif "should" in text:
            return f"∃x, {text.replace('should', '⇒')}"
        else:
            return f"⟂{text}"

    def _mine_assumptions(
        self,
        prompt_input: PromptInput,
        constraints: List[Constraint]
    ) -> List[Dict[str, Any]]:
        """
        Mine tacit assumptions using Φ₁.₅.

        Args:
            prompt_input: Input prompt
            constraints: Extracted constraints

        Returns:
            List of mined assumptions
        """
        assumptions = []

        # Create null result from constraints (simulated)
        # In production, this would come from actual failures

        for i, constraint in enumerate(constraints):
            # Simulate potential failure scenarios
            assumption = {
                'id': f'assumption_{i}',
                'type': AssumptionType.CONSTRAINT.value,
                'description': f"Assumption: {constraint.description} is feasible",
                'confidence': 0.7,
                'source_constraint_id': constraint.id,
                'rationale': "Constraint may have hidden feasibility assumptions"
            }
            assumptions.append(assumption)

        return assumptions

    def _detect_biases(
        self,
        prompt_input: PromptInput,
        constraints: List[Constraint]
    ) -> Dict[str, Any]:
        """
        Detect cognitive biases using Φ₂.

        Args:
            prompt_input: Input prompt
            constraints: Extracted constraints

        Returns:
            Bias detection report
        """
        # Use cognitive bias detector
        bias_report = self.phi2.analyze_constraints(constraints)

        return {
            'overall_bias_score': bias_report.overall_bias_score,
            'total_detections': bias_report.total_detections,
            'bias_types': [
                {
                    'type': bias.bias_type.value,
                    'severity': bias.severity.value,
                    'confidence': bias.confidence,
                    'description': bias.description,
                    'affected_elements': bias.affected_elements if hasattr(bias, 'affected_elements') else []
                }
                for bias in bias_report.detections
            ]
        }

    def _feedback_loop_refinement(
        self,
        prompt_input: PromptInput,
        result: PromptAnalysisResult,
        iterations: int = 2
    ) -> PromptAnalysisResult:
        """
        Φ₁.₅ feedback loop for prompt refinement.

        Args:
            prompt_input: Original input prompt
            result: Current analysis result
            iterations: Number of refinement iterations

        Returns:
            Refined analysis result
        """
        result.status = PromptAnalysisStatus.REFINING

        for iteration in range(iterations):
            # Use assumptions to refine constraints
            refined_constraints = []

            for constraint in result.constraints:
                # Check if any assumptions challenge this constraint
                challenged = False

                for assumption in result.assumptions:
                    if assumption.get('source_constraint_id') == constraint.id:
                        if assumption.get('confidence', 0.0) > 0.7:
                            challenged = True
                            break

                if not challenged:
                    refined_constraints.append(constraint)

            # Update result
            result.constraints = refined_constraints

            # Check for convergence
            if len(result.assumptions) == 0:
                break

        return result

    def _calculate_confidence(
        self,
        result: PromptAnalysisResult
    ) -> float:
        """
        Calculate overall confidence score.

        Args:
            result: Analysis result

        Returns:
            Confidence score [0, 1]
        """
        # Base confidence from constraints
        if len(result.constraints) == 0:
            base_confidence = 0.3
        else:
            base_confidence = 0.7

        # Adjust based on assumptions (more assumptions = lower confidence)
        assumption_penalty = min(len(result.assumptions) * 0.05, 0.3)

        # Adjust based on biases (more biases = lower confidence)
        bias_penalty = 0.0
        if result.bias_report:
            bias_penalty = min(result.bias_report.get('total_detections', 0) * 0.1, 0.3)

        confidence = base_confidence - assumption_penalty - bias_penalty

        return max(0.0, min(1.0, confidence))

    def get_sce_state(self) -> Dict[str, Any]:
        """
        Get current SCE state.

        Returns:
            SCE state dictionary
        """
        if not self.enable_sce:
            return {'enabled': False}

        constraints = self.sce.get_all_constraints()

        return {
            'enabled': True,
            'total_constraints': len(constraints),
            'hard_constraints': len([c for c in constraints if c.is_hard()]),
            'soft_constraints': len([c for c in constraints if c.type == ConstraintType.SOFT]),
            'preference_constraints': len([c for c in constraints if c.type == ConstraintType.PREFERENCE]),
            'verified_constraints': len([c for c in constraints if c.is_verified()])
        }

    def export_analysis(
        self,
        result: PromptAnalysisResult,
        output_path: Optional[Path] = None
    ) -> Path:
        """
        Export analysis result to JSON.

        Args:
            result: Analysis result to export
            output_path: Optional output path

        Returns:
            Path to exported file
        """
        if output_path is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_path = Path(f"stage1_analysis_{timestamp}.json")

        with open(output_path, 'w') as f:
            json.dump(result.to_dict(), f, indent=2)

        return output_path


# ============================================================================
# Convenience Functions
# ============================================================================

def analyze_prompt(
    prompt_text: str,
    domain: str = "general",
    config: Optional[Dict[str, Any]] = None
) -> PromptAnalysisResult:
    """
    Convenience function to analyze a prompt.

    Args:
        prompt_text: Prompt text to analyze
        domain: Problem domain
        config: Optional configuration

    Returns:
        PromptAnalysisResult
    """
    integration = Stage1Integration(config=config)

    prompt_input = PromptInput(
        text=prompt_text,
        domain=domain
    )

    return integration.analyze_prompt(prompt_input)


# ============================================================================
# Module Exports
# ============================================================================

__all__ = [
    # Main class
    'Stage1Integration',

    # Data structures
    'PromptInput',
    'PromptAnalysisResult',
    'ConstraintExtraction',

    # Enums
    'PromptAnalysisStatus',

    # Convenience functions
    'analyze_prompt',
]
