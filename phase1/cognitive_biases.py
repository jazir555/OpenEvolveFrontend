"""
Phi 2 Metacognitive Debiasing System

Detects and mitigates cognitive biases in problem formulation,
constraint specification, and solution generation.

Author: Agent B2 (Phi 2 Specialist)
Created: 2025-12-31
Status: Green - Active Implementation
Phase: Phase I - Epistemic Audit
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Callable
from enum import Enum
import re
from collections import Counter
import math

# Import SCE for integration
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "core"))
from symbolic_constraint_engine import Constraint, ConstraintType


class BiasType(Enum):
    """Types of cognitive biases detected by Φ₂"""
    CONFIRMATION = "confirmation_bias"
    AVAILABILITY = "availability_bias"
    ANCHORING = "anchoring_bias"
    SUNK_COST = "sunk_cost_fallacy"
    FRAMING = "framing_effect"
    OVERCONFIDENCE = "overconfidence_effect"
    DUNNING_KRUGER = "dunning_kruger_effect"
    AUTHORITY = "authority_bias"
    CLUSTERING = "clustering_illusion"
    TEXAS_SHARPSHOOTER = "texas_sharpshooter_fallacy"
    CAUSAL_OVERSIMPLIFICATION = "causal_oversimplification"
    ILLUSION_OF_CONTROL = "illusion_of_control"


class Severity(Enum):
    """Severity levels for detected biases"""
    LOW = 1      # Minor bias, optional intervention
    MEDIUM = 2   # Moderate bias, recommended intervention
    HIGH = 3     # Severe bias, required intervention
    CRITICAL = 4 # Extreme bias, immediate intervention required


@dataclass
class BiasDetection:
    """
    A detected cognitive bias instance.

    Attributes:
        bias_type: Type of bias detected
        severity: Severity level (1-4)
        confidence: Detector confidence [0, 1]
        description: Human-readable description
        evidence: Evidence supporting the detection
        suggestion: Suggested debiasing intervention
        affected_elements: List of affected constraint IDs or solution components
    """
    bias_type: BiasType
    severity: Severity
    confidence: float
    description: str
    evidence: Dict[str, str] = field(default_factory=dict)
    suggestion: str = ""
    affected_elements: List[str] = field(default_factory=list)

    def __post_init__(self):
        """Validate detection after initialization"""
        if not 0 <= self.confidence <= 1:
            raise ValueError(f"Confidence must be in [0, 1], got {self.confidence}")
        if self.severity not in Severity:
            raise ValueError(f"Invalid severity: {self.severity}")

    def is_critical(self) -> bool:
        """Check if this is a critical bias requiring immediate intervention"""
        return self.severity in [Severity.HIGH, Severity.CRITICAL]

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        return {
            "bias_type": self.bias_type.value,
            "severity": self.severity.value,
            "confidence": self.confidence,
            "description": self.description,
            "evidence": self.evidence,
            "suggestion": self.suggestion,
            "affected_elements": self.affected_elements
        }


@dataclass
class BiasReport:
    """
    Comprehensive bias analysis report.

    Attributes:
        total_detections: Total number of biases detected
        detections_by_type: Breakdown by bias type
        detections_by_severity: Breakdown by severity
        overall_bias_score: Aggregate bias score [0, 1]
        recommendations: Prioritized recommendations
    """
    total_detections: int = 0
    detections_by_type: Dict[BiasType, int] = field(default_factory=dict)
    detections_by_severity: Dict[Severity, int] = field(default_factory=dict)
    overall_bias_score: float = 0.0
    recommendations: List[str] = field(default_factory=list)
    detections: List[BiasDetection] = field(default_factory=list)

    def add_detection(self, detection: BiasDetection) -> None:
        """Add a detection to the report"""
        self.detections.append(detection)
        self.total_detections += 1

        # Update by type
        if detection.bias_type not in self.detections_by_type:
            self.detections_by_type[detection.bias_type] = 0
        self.detections_by_type[detection.bias_type] += 1

        # Update by severity
        if detection.severity not in self.detections_by_severity:
            self.detections_by_severity[detection.severity] = 0
        self.detections_by_severity[detection.severity] += 1

    def calculate_overall_score(self) -> None:
        """Calculate overall bias score from detections"""
        if not self.detections:
            self.overall_bias_score = 0.0
            return

        # Weight by severity and confidence
        weighted_sum = 0.0
        for detection in self.detections:
            severity_weight = detection.severity.value / 4.0  # Normalize to [0, 1]
            weighted_sum += severity_weight * detection.confidence

        # Normalize by number of detections (cap at 1.0)
        self.overall_bias_score = min(weighted_sum / len(self.detections), 1.0)


class CognitiveBiasDetector:
    """
    Main class for detecting cognitive biases in the RESE system.

    This class orchestrates multiple specialized bias detectors and
    provides a unified interface for bias detection and mitigation.
    """

    def __init__(self):
        self.detection_history: List[BiasReport] = []
        self.detectors: Dict[BiasType, Callable] = self._initialize_detectors()

    def _initialize_detectors(self) -> Dict[BiasType, Callable]:
        """Initialize all bias detector functions"""
        return {
            BiasType.CONFIRMATION: self._detect_confirmation_bias,
            BiasType.AVAILABILITY: self._detect_availability_bias,
            BiasType.ANCHORING: self._detect_anchoring_bias,
            BiasType.SUNK_COST: self._detect_sunk_cost_fallacy,
            BiasType.FRAMING: self._detect_framing_effect,
            BiasType.OVERCONFIDENCE: self._detect_overconfidence_effect,
            BiasType.DUNNING_KRUGER: self._detect_dunning_kruger_effect,
            BiasType.AUTHORITY: self._detect_authority_bias,
            BiasType.CLUSTERING: self._detect_clustering_illusion,
            BiasType.TEXAS_SHARPSHOOTER: self._detect_texas_sharpshooter_fallacy,
            BiasType.CAUSAL_OVERSIMPLIFICATION: self._detect_causal_oversimplification,
            BiasType.ILLUSION_OF_CONTROL: self._detect_illusion_of_control,
        }

    # ========================================
    # TEXT-BASED BIAS DETECTORS
    # ========================================

    def _detect_confirmation_bias(
        self,
        constraints: List[Constraint],
        context: Optional[Dict] = None
    ) -> List[BiasDetection]:
        """
        Detect confirmation bias in constraint formulation.

        Indicators:
        - Absolutist language ("clearly", "obviously", "undoubtedly")
        - One-sided evidence presentation
        - Lack of alternative hypotheses
        - Selective citation patterns
        """
        detections = []

        absolutist_terms = [
            "clearly", "obviously", "undoubtedly", "certainly", "definitely",
            "absolutely", "unquestionably", "surely", "necessarily"
        ]

        confirming_terms = [
            "confirms", "shows", "proves", "demonstrates", "validates"
        ]

        for constraint in constraints:
            desc_lower = constraint.description.lower()
            form_lower = constraint.formalization.lower()

            # Check for absolutist language
            absolutist_count = sum(1 for term in absolutist_terms if term in desc_lower)

            # Check for one-sided confirming language
            confirming_count = sum(1 for term in confirming_terms if term in desc_lower)

            # Check for lack of alternative consideration
            has_alternatives = any([
                "however" in desc_lower,
                "although" in desc_lower,
                "conversely" in desc_lower,
                "alternatively" in desc_lower,
                "on the other hand" in desc_lower
            ])

            # Calculate confidence based on indicators
            confidence = 0.0
            evidence = {}

            if absolutist_count > 0:
                confidence += 0.3 * min(absolutist_count, 3) / 3
                evidence["absolutist_terms"] = f"Found {absolutist_count} absolutist terms"

            if confirming_count > 0 and not has_alternatives:
                confidence += 0.4
                evidence["one_sided"] = "Only confirming language, no alternatives considered"

            if not has_alternatives:
                confidence += 0.3
                evidence["no_alternatives"] = "No alternative hypotheses considered"

            # Determine severity based on confidence and constraint type
            if confidence > 0.7:
                severity = Severity.HIGH
            elif confidence > 0.5:
                severity = Severity.MEDIUM
            elif confidence > 0.3:
                severity = Severity.LOW
            else:
                continue  # Below threshold

            detection = BiasDetection(
                bias_type=BiasType.CONFIRMATION,
                severity=severity,
                confidence=confidence,
                description=f"Potential confirmation bias in constraint '{constraint.id}'",
                evidence=evidence,
                suggestion="Consider alternative hypotheses and seek disconfirming evidence",
                affected_elements=[constraint.id]
            )
            detections.append(detection)

        return detections

    def _detect_overconfidence_effect(
        self,
        constraints: List[Constraint],
        context: Optional[Dict] = None
    ) -> List[BiasDetection]:
        """
        Detect overconfidence in constraint specification.

        Indicators:
        - Point estimates without uncertainty ranges
        - High confidence claims without qualification
        - Narrow prediction intervals
        - Absence of "may", "might", "could", "approximately"
        """
        detections = []

        certain_terms = [
            "will", "shall", "must", "always", "never", "certainly",
            "definitely", "precisely", "exactly", "without fail"
        ]

        uncertain_terms = [
            "may", "might", "could", "approximately", "roughly",
            "likely", "probably", "possibly", "suggests", "indicates"
        ]

        for constraint in constraints:
            desc_lower = constraint.description.lower()

            # Count certain vs. uncertain terms
            certain_count = sum(1 for term in certain_terms if term in desc_lower)
            uncertain_count = sum(1 for term in uncertain_terms if term in desc_lower)

            # Check for point estimates (e.g., "100%", "exactly 10", "0 error")
            point_estimate_patterns = [
                r'\d+%',  # Exact percentages
                r'exactly\s+\d+',  # Exactly followed by number
                r'precisely\s+\d+',  # Precisely followed by number
                r'zero\s+error',  # Zero error claims
                r'no\s+uncertainty',  # No uncertainty claims
            ]

            point_estimate_count = 0
            for pattern in point_estimate_patterns:
                if re.search(pattern, desc_lower):
                    point_estimate_count += 1

            # Calculate confidence
            confidence = 0.0
            evidence = {}

            if certain_count > uncertain_count:
                confidence += 0.4
                evidence["certain_language"] = f"Uses {certain_count} certain terms vs {uncertain_count} uncertain terms"

            if point_estimate_count > 0:
                confidence += 0.3 * min(point_estimate_count, 3) / 3
                evidence["point_estimates"] = f"Found {point_estimate_count} point estimate patterns"

            if uncertain_count == 0:
                confidence += 0.3
                evidence["no_uncertainty"] = "No acknowledgment of uncertainty"

            # Severity determination
            if confidence > 0.7:
                severity = Severity.HIGH
            elif confidence > 0.5:
                severity = Severity.MEDIUM
            elif confidence > 0.3:
                severity = Severity.LOW
            else:
                continue

            detection = BiasDetection(
                bias_type=BiasType.OVERCONFIDENCE,
                severity=severity,
                confidence=confidence,
                description=f"Potential overconfidence in constraint '{constraint.id}'",
                evidence=evidence,
                suggestion="Include uncertainty quantification and confidence intervals",
                affected_elements=[constraint.id]
            )
            detections.append(detection)

        return detections

    def _detect_framing_effect(
        self,
        constraints: List[Constraint],
        context: Optional[Dict] = None
    ) -> List[BiasDetection]:
        """
        Detect framing effects in constraint formulation.

        Indicators:
        - Loss vs. gain framing
        - Emotional language
        - Positive vs. negative wording influence
        """
        detections = []

        # Loss frame terms
        loss_terms = [
            "avoid", "prevent", "reduce", "minimize", "mitigate",
            "loss", "risk", "threat", "danger", "failure"
        ]

        # Gain frame terms
        gain_terms = [
            "achieve", "gain", "obtain", "maximize", "enhance",
            "benefit", "improve", "increase", "success", "opportunity"
        ]

        for constraint in constraints:
            desc_lower = constraint.description.lower()

            loss_count = sum(1 for term in loss_terms if term in desc_lower)
            gain_count = sum(1 for term in gain_terms if term in desc_lower)

            # Check for emotional language
            emotional_terms = [
                "urgent", "critical", "disaster", "crisis", "emergency",
                "wonderful", "excellent", "terrible", "horrible", "devastating"
            ]
            emotional_count = sum(1 for term in emotional_terms if term in desc_lower)

            confidence = 0.0
            evidence = {}

            # Strong framing bias if heavily skewed
            if loss_count > 2 and gain_count == 0:
                confidence += 0.5
                evidence["loss_frame"] = f"Heavily loss-framed ({loss_count} loss terms, 0 gain terms)"
            elif gain_count > 2 and loss_count == 0:
                confidence += 0.5
                evidence["gain_frame"] = f"Heavily gain-framed ({gain_count} gain terms, 0 loss terms)"

            if emotional_count > 0:
                confidence += 0.3
                evidence["emotional_language"] = f"Found {emotional_count} emotional terms"

            # Severity
            if confidence > 0.6:
                severity = Severity.MEDIUM
            elif confidence > 0.3:
                severity = Severity.LOW
            else:
                continue

            detection = BiasDetection(
                bias_type=BiasType.FRAMING,
                severity=severity,
                confidence=confidence,
                description=f"Potential framing effect in constraint '{constraint.id}'",
                evidence=evidence,
                suggestion="Consider reframing with neutral language to test robustness",
                affected_elements=[constraint.id]
            )
            detections.append(detection)

        return detections

    def _detect_authority_bias(
        self,
        constraints: List[Constraint],
        context: Optional[Dict] = None
    ) -> List[BiasDetection]:
        """
        Detect authority bias in constraint sources.

        Indicators:
        - Excessive citations from authoritative sources
        - Neglect of non-authoritative but valid evidence
        - Appeals to authority rather than evidence quality
        """
        detections = []

        authority_indicators = [
            "expert", "authority", "professor", "dr.", "ph.d",
            "study shows", "research proves", "published in",
            "peer-reviewed", "established", "accepted"
        ]

        for constraint in constraints:
            source_lower = constraint.source.lower()
            desc_lower = constraint.description.lower()

            # Count authority indicators
            authority_count = sum(1 for term in authority_indicators if term in source_lower)
            authority_in_desc = sum(1 for term in authority_indicators if term in desc_lower)

            # Check if justification is purely authority-based
            pure_authority = authority_count > 0 and "evidence" not in desc_lower.lower()

            confidence = 0.0
            evidence = {}

            if authority_count > 0:
                confidence += 0.4
                evidence["authority_citations"] = f"Source contains {authority_count} authority indicators"

            if authority_in_desc > 0:
                confidence += 0.3
                evidence["authority_in_description"] = f"Description contains {authority_in_desc} authority indicators"

            if pure_authority:
                confidence += 0.3
                evidence["pure_authority"] = "Justification based on authority, not evidence"

            # Severity
            if confidence > 0.6:
                severity = Severity.MEDIUM
            elif confidence > 0.3:
                severity = Severity.LOW
            else:
                continue

            detection = BiasDetection(
                bias_type=BiasType.AUTHORITY,
                severity=severity,
                confidence=confidence,
                description=f"Potential authority bias in constraint '{constraint.id}'",
                evidence=evidence,
                suggestion="Evaluate evidence quality independently of source authority",
                affected_elements=[constraint.id]
            )
            detections.append(detection)

        return detections

    # ========================================
    # STRUCTURAL BIAS DETECTORS
    # ========================================

    def _detect_anchoring_bias(
        self,
        constraints: List[Constraint],
        context: Optional[Dict] = None
    ) -> List[BiasDetection]:
        """
        Detect anchoring bias in constraint formulation sequence.

        Indicators:
        - Early constraints heavily referenced by later constraints
        - Star topology around initial constraints
        - Low variance in constraint additions over time
        """
        detections = []

        if not constraints or len(constraints) < 3:
            return detections

        # Analyze dependency patterns (if dependency info available)
        # For now, analyze temporal patterns based on constraint IDs or indices

        # Simple heuristic: check if early constraints have similar
        # formulations to later constraints (potential anchoring)

        early_constraints = constraints[:len(constraints)//3]
        late_constraints = constraints[len(constraints)//3:]

        if not early_constraints or not late_constraints:
            return detections

        # Check for similarity in formulation patterns
        early_formalizations = [c.formalization.lower() for c in early_constraints]
        late_formalizations = [c.formalization.lower() for c in late_constraints]

        # Extract common patterns
        early_words = Counter()
        for form in early_formalizations:
            words = re.findall(r'\w+', form)
            early_words.update(words)

        late_words = Counter()
        for form in late_formalizations:
            words = re.findall(r'\w+', form)
            late_words.update(words)

        # Calculate overlap
        common_words = set(early_words.keys()) & set(late_words.keys())
        overlap_ratio = len(common_words) / max(len(set(early_words.keys())), 1)

        confidence = 0.0
        evidence = {}

        if overlap_ratio > 0.5:
            confidence += 0.5
            evidence["high_overlap"] = f"High formulation overlap ({overlap_ratio:.2%}) between early and late constraints"

        # Check for constraint density (star topology indicator)
        # If one constraint is referenced by many others
        # (Simplified: we'd need actual dependency graph for this)

        # Severity
        if confidence > 0.4:
            severity = Severity.MEDIUM
        elif confidence > 0.2:
            severity = Severity.LOW
        else:
            return detections

        detection = BiasDetection(
            bias_type=BiasType.ANCHORING,
            severity=severity,
            confidence=confidence,
            description="Potential anchoring bias: later constraints may be anchored to early formulations",
            evidence=evidence,
            suggestion="Generate multiple independent initial formulations and compare",
            affected_elements=[c.id for c in constraints[:3]]
        )
        detections.append(detection)

        return detections

    def _detect_availability_bias(
        self,
        constraints: List[Constraint],
        context: Optional[Dict] = None
    ) -> List[BiasDetection]:
        """
        Detect availability bias in constraint sources.

        Indicators:
        - Skewed distribution of constraint sources
        - Over-representation of familiar domains
        - Lack of domain diversity
        """
        detections = []

        if not constraints:
            return detections

        # Analyze source distribution
        sources = [c.source for c in constraints]
        source_counts = Counter(sources)

        # Calculate entropy of source distribution
        total = len(sources)
        entropy = 0.0
        for count in source_counts.values():
            if count > 0:
                p = count / total
                entropy -= p * math.log2(p)

        # Normalize entropy (max is log2 of number of unique sources)
        max_entropy = math.log2(len(source_counts)) if len(source_counts) > 1 else 1
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0

        confidence = 0.0
        evidence = {}

        # Low entropy indicates concentration (potential availability bias)
        if normalized_entropy < 0.5:
            confidence += 0.5
            evidence["low_source_diversity"] = f"Low source diversity (normalized entropy: {normalized_entropy:.2f})"
            evidence["source_distribution"] = dict(source_counts.most_common(3))

        # Check for a single dominant source (>50% of constraints)
        if source_counts:
            top_source_ratio = source_counts.most_common(1)[0][1] / total
            if top_source_ratio > 0.5:
                confidence += 0.3
                evidence["dominant_source"] = f"Single source provides {top_source_ratio:.1%} of constraints"

        # Severity
        if confidence > 0.5:
            severity = Severity.HIGH
        elif confidence > 0.3:
            severity = Severity.MEDIUM
        elif confidence > 0.1:
            severity = Severity.LOW
        else:
            return detections

        detection = BiasDetection(
            bias_type=BiasType.AVAILABILITY,
            severity=severity,
            confidence=confidence,
            description="Potential availability bias: constraint sources may be skewed by availability",
            evidence=evidence,
            suggestion="Diversify constraint sources and seek unfamiliar domains",
            affected_elements=list(set([c.id for c in constraints]))
        )
        detections.append(detection)

        return detections

    def _detect_clustering_illusion(
        self,
        constraints: List[Constraint],
        context: Optional[Dict] = None
    ) -> List[BiasDetection]:
        """
        Detect clustering illusion in constraint relationships.

        Indicators:
        - Seeing patterns in random constraint groupings
        - Over-interpreting coincidental correlations
        - Assuming causal relationships from mere correlations
        """
        detections = []

        # Look for causal language in constraints
        causal_indicators = [
            "causes", "because of", "due to", "leads to", "results in",
            "therefore", "thus", "consequently", "as a result"
        ]

        for constraint in constraints:
            desc_lower = constraint.description.lower()
            form_lower = constraint.formalization.lower()

            causal_count = sum(1 for term in causal_indicators if term in desc_lower or term in form_lower)

            # Check for statistical qualifiers (absence suggests illusion)
            has_stats = any([
                "correlation" in desc_lower,
                "significant" in desc_lower,
                "p-value" in desc_lower,
                "confidence interval" in desc_lower,
                "statistically" in desc_lower
            ])

            confidence = 0.0
            evidence = {}

            if causal_count > 0 and not has_stats:
                confidence += 0.6
                evidence["causal_without_stats"] = f"Causal language used ({causal_count} instances) without statistical validation"

            # Check for pattern-detection language
            pattern_terms = ["pattern", "trend", "clearly shows", "obviously"]
            pattern_count = sum(1 for term in pattern_terms if term in desc_lower)
            if pattern_count > 0 and not has_stats:
                confidence += 0.4
                evidence["pattern_without_stats"] = f"Pattern claims ({pattern_count}) without statistical testing"

            # Severity
            if confidence > 0.6:
                severity = Severity.HIGH
            elif confidence > 0.3:
                severity = Severity.MEDIUM
            elif confidence > 0.1:
                severity = Severity.LOW
            else:
                continue

            detection = BiasDetection(
                bias_type=BiasType.CLUSTERING,
                severity=severity,
                confidence=confidence,
                description=f"Potential clustering illusion in constraint '{constraint.id}'",
                evidence=evidence,
                suggestion="Apply statistical significance testing and consider null hypotheses",
                affected_elements=[constraint.id]
            )
            detections.append(detection)

        return detections

    # ========================================
    # BEHAVIORAL BIAS DETECTORS
    # ========================================

    def _detect_sunk_cost_fallacy(
        self,
        constraints: List[Constraint],
        context: Optional[Dict] = None
    ) -> List[BiasDetection]:
        """
        Detect sunk cost fallacy in constraint persistence.

        Indicators:
        - Persistence with failing approaches
        - Reluctance to abandon complex constraint sets
        - Path dependency in constraint evolution

        Note: This requires historical data about constraint changes
        """
        detections = []

        # Without historical data, we can only look for linguistic markers
        # of sunk cost reasoning

        sunk_cost_indicators = [
            "we've already", "we have invested", "spent time on",
            "already done", "existing work", "previous effort"
        ]

        for constraint in constraints:
            desc_lower = constraint.description.lower()

            sunk_cost_count = sum(1 for term in sunk_cost_indicators if term in desc_lower)

            if sunk_cost_count > 0:
                confidence = min(0.7, 0.3 + sunk_cost_count * 0.2)
                evidence = {
                    "sunk_cost_language": f"Found {sunk_cost_count} sunk cost indicators"
                }

                severity = Severity.MEDIUM if confidence > 0.5 else Severity.LOW

                detection = BiasDetection(
                    bias_type=BiasType.SUNK_COST,
                    severity=severity,
                    confidence=confidence,
                    description=f"Potential sunk cost fallacy in constraint '{constraint.id}'",
                    evidence=evidence,
                    suggestion="Evaluate constraints independently of prior investment (zero-based costing)",
                    affected_elements=[constraint.id]
                )
                detections.append(detection)

        return detections

    def _detect_texas_sharpshooter_fallacy(
        self,
        constraints: List[Constraint],
        context: Optional[Dict] = None
    ) -> List[BiasDetection]:
        """
        Detect Texas sharpshooter fallacy in constraint formulation.

        Indicators:
        - Post-hoc constraint selection
        - Cherry-picking data to support conclusions
        - Hindsight bias in explanations
        """
        detections = []

        post_hoc_indicators = [
            "in retrospect", "in hindsight", "looking back",
            "as we can now see", "it is clear now that",
            "post-hoc", "after the fact"
        ]

        for constraint in constraints:
            desc_lower = constraint.description.lower()

            post_hoc_count = sum(1 for term in post_hoc_indicators if term in desc_lower)

            # Check for narrative fallacy (creating coherent stories post-hoc)
            narrative_indicators = [
                "story", "narrative", "explains why", "the reason is"
            ]
            narrative_count = sum(1 for term in narrative_indicators if term in desc_lower)

            confidence = 0.0
            evidence = {}

            if post_hoc_count > 0:
                confidence += 0.5
                evidence["post_hoc_language"] = f"Found {post_hoc_count} post-hoc indicators"

            if narrative_count > 0:
                confidence += 0.3
                evidence["narrative_fallacy"] = "Potential narrative fallacy (coherent story constructed)"

            # Severity
            if confidence > 0.5:
                severity = Severity.MEDIUM
            elif confidence > 0.2:
                severity = Severity.LOW
            else:
                continue

            detection = BiasDetection(
                bias_type=BiasType.TEXAS_SHARPSHOOTER,
                severity=severity,
                confidence=confidence,
                description=f"Potential Texas sharpshooter fallacy in constraint '{constraint.id}'",
                evidence=evidence,
                suggestion="Pre-register hypotheses and evaluate against independent data",
                affected_elements=[constraint.id]
            )
            detections.append(detection)

        return detections

    def _detect_dunning_kruger_effect(
        self,
        constraints: List[Constraint],
        context: Optional[Dict] = None
    ) -> List[BiasDetection]:
        """
        Detect Dunning-Kruger effect in constraint formulation.

        Indicators:
        - High confidence with low complexity
        - Oversimplified formulations
        - Lack of acknowledgment of limitations
        """
        detections = []

        for constraint in constraints:
            desc_lower = constraint.description.lower()

            # Check for complexity (simple formulations may indicate lack of expertise)
            word_count = len(re.findall(r'\w+', constraint.description))
            unique_words = len(set(re.findall(r'\w+', constraint.description.lower())))
            lexical_diversity = unique_words / max(word_count, 1)

            # Check for confidence indicators
            confident_terms = ["easy", "simple", "trivial", "obvious", "straightforward"]
            confident_count = sum(1 for term in confident_terms if term in desc_lower)

            # Check for acknowledgment of limitations
            limitation_terms = ["limitation", "constraint", "assumption", " caveat", "however"]
            has_limitations = any(term in desc_lower for term in limitation_terms)

            confidence = 0.0
            evidence = {}

            # High confidence + low complexity + no limitations = potential D-K
            if confident_count > 0 and lexical_diversity < 0.5 and not has_limitations:
                confidence += 0.7
                evidence["high_confidence_low_complexity"] = f"Confident language ({confident_count} terms) with low lexical diversity ({lexical_diversity:.2f}) and no acknowledged limitations"

            # Severity
            if confidence > 0.6:
                severity = Severity.HIGH
            elif confidence > 0.3:
                severity = Severity.MEDIUM
            elif confidence > 0.1:
                severity = Severity.LOW
            else:
                continue

            detection = BiasDetection(
                bias_type=BiasType.DUNNING_KRUGER,
                severity=severity,
                confidence=confidence,
                description=f"Potential Dunning-Kruger effect in constraint '{constraint.id}'",
                evidence=evidence,
                suggestion="Seek expert review and explicitly acknowledge limitations",
                affected_elements=[constraint.id]
            )
            detections.append(detection)

        return detections

    def _detect_causal_oversimplification(
        self,
        constraints: List[Constraint],
        context: Optional[Dict] = None
    ) -> List[BiasDetection]:
        """
        Detect causal oversimplification in constraints.

        Indicators:
        - Single-cause explanations for complex phenomena
        - Neglect of interaction effects
        - Linear assumptions in non-linear systems
        """
        detections = []

        for constraint in constraints:
            desc_lower = constraint.description.lower()
            form_lower = constraint.formalization.lower()

            # Check for single-cause language
            single_cause_patterns = [
                r'is caused by',
                r'is due to',
                r'the cause is',
                r'solely because',
                r'only because'
            ]

            single_cause_count = 0
            for pattern in single_cause_patterns:
                if re.search(pattern, desc_lower):
                    single_cause_count += 1

            # Check for multi-factor acknowledgment
            multi_factor_terms = [
                "multiple factors", "several causes", "interaction",
                "complex", "multifaceted", "interplay"
            ]
            has_multi_factor = any(term in desc_lower for term in multi_factor_terms)

            # Check for linearity assumptions
            linear_terms = ["linear", "proportional", "directly related"]
            has_linear = any(term in desc_lower for term in linear_terms)

            confidence = 0.0
            evidence = {}

            if single_cause_count > 0 and not has_multi_factor:
                confidence += 0.5
                evidence["single_cause"] = f"Single-cause language ({single_cause_count} instances) without multi-factor acknowledgment"

            if has_linear and not has_multi_factor:
                confidence += 0.3
                evidence["linear_assumption"] = "Linear assumption without considering interactions"

            # Severity
            if confidence > 0.5:
                severity = Severity.MEDIUM
            elif confidence > 0.2:
                severity = Severity.LOW
            else:
                continue

            detection = BiasDetection(
                bias_type=BiasType.CAUSAL_OVERSIMPLIFICATION,
                severity=severity,
                confidence=confidence,
                description=f"Potential causal oversimplification in constraint '{constraint.id}'",
                evidence=evidence,
                suggestion="Consider multiple causal factors and interaction effects",
                affected_elements=[constraint.id]
            )
            detections.append(detection)

        return detections

    def _detect_illusion_of_control(
        self,
        constraints: List[Constraint],
        context: Optional[Dict] = None
    ) -> List[BiasDetection]:
        """
        Detect illusion of control in constraint formulation.

        Indicators:
        - Assuming deterministic control over stochastic processes
        - Underestimating external factors
        - Over-precision in predictions
        """
        detections = []

        for constraint in constraints:
            desc_lower = constraint.description.lower()
            form_lower = constraint.formalization.lower()

            # Check for deterministic language
            deterministic_terms = [
                "will", "shall", "must", "always", "never",
                "certainly", "definitely", "precisely", "exactly"
            ]
            deterministic_count = sum(1 for term in deterministic_terms if term in desc_lower)

            # Check for acknowledgment of external factors
            external_factor_terms = [
                "external", "outside", "uncontrollable", "random",
                "stochastic", "uncertain", "unpredictable"
            ]
            has_external = any(term in desc_lower for term in external_factor_terms)

            # Check for probability/uncertainty language
            uncertainty_terms = [
                "probability", "likelihood", "chance", "risk",
                "uncertainty", "maybe", "might", "could"
            ]
            has_uncertainty = any(term in desc_lower for term in uncertainty_terms)

            confidence = 0.0
            evidence = {}

            if deterministic_count > 0 and not has_uncertainty:
                confidence += 0.4
                evidence["deterministic_without_uncertainty"] = f"Deterministic language ({deterministic_count} terms) without uncertainty acknowledgment"

            if not has_external:
                confidence += 0.3
                evidence["no_external_factors"] = "No acknowledgment of external or uncontrollable factors"

            if not has_uncertainty:
                confidence += 0.3
                evidence["no_uncertainty"] = "No probabilistic or stochastic language"

            # Severity
            if confidence > 0.6:
                severity = Severity.HIGH
            elif confidence > 0.4:
                severity = Severity.MEDIUM
            elif confidence > 0.2:
                severity = Severity.LOW
            else:
                continue

            detection = BiasDetection(
                bias_type=BiasType.ILLUSION_OF_CONTROL,
                severity=severity,
                confidence=confidence,
                description=f"Potential illusion of control in constraint '{constraint.id}'",
                evidence=evidence,
                suggestion="Explicitly model uncertainty and external factors",
                affected_elements=[constraint.id]
            )
            detections.append(detection)

        return detections

    # ========================================
    # PUBLIC API
    # ========================================

    def analyze_constraints(
        self,
        constraints: List[Constraint],
        bias_types: Optional[List[BiasType]] = None
    ) -> BiasReport:
        """
        Analyze a list of constraints for cognitive biases.

        Args:
            constraints: List of constraints to analyze
            bias_types: Optional list of specific bias types to detect.
                       If None, detects all bias types.

        Returns:
            BiasReport with all detections and overall score
        """
        report = BiasReport()

        # Determine which detectors to run
        detectors_to_run = (
            [self.detectors[bt] for bt in bias_types]
            if bias_types
            else list(self.detectors.values())
        )

        # Run all detectors
        all_detections = []
        for detector in detectors_to_run:
            try:
                detections = detector(constraints)
                all_detections.extend(detections)
            except Exception as e:
                print(f"Warning: Detector {detector.__name__} failed: {e}")
                continue

        # Add detections to report
        for detection in all_detections:
            report.add_detection(detection)

        # Calculate overall score
        report.calculate_overall_score()

        # Generate recommendations
        report.recommendations = self._generate_recommendations(report)

        # Store report in history
        self.detection_history.append(report)

        return report

    def _generate_recommendations(self, report: BiasReport) -> List[str]:
        """Generate prioritized recommendations based on report"""
        recommendations = []

        # Sort detections by severity (high to low) and confidence (high to low)
        sorted_detections = sorted(
            report.detections,
            key=lambda d: (d.severity.value, d.confidence),
            reverse=True
        )

        # Top priority: Critical and High severity biases
        critical_high = [d for d in sorted_detections if d.is_critical()]
        if critical_high:
            recommendations.append(
                f"URGENT: Address {len(critical_high)} critical/high-severity biases immediately"
            )
            for detection in critical_high[:3]:  # Top 3
                recommendations.append(
                    f"  - {detection.bias_type.value}: {detection.suggestion}"
                )

        # Medium priority: Medium severity biases
        medium = [d for d in sorted_detections if d.severity == Severity.MEDIUM]
        if medium:
            recommendations.append(
                f"IMPORTANT: Review {len(medium)} medium-severity biases"
            )
            # Group by bias type
            by_type = {}
            for d in medium:
                if d.bias_type not in by_type:
                    by_type[d.bias_type] = []
                by_type[d.bias_type].append(d)
            for bias_type, dets in list(by_type.items())[:3]:
                recommendations.append(
                    f"  - {bias_type.value}: {len(dets)} instances"
                )

        # Low priority: Low severity biases
        low = [d for d in sorted_detections if d.severity == Severity.LOW]
        if len(low) > 5:
            recommendations.append(
                f"Note: {len(low)} low-severity biases detected (consider addressing if time permits)"
            )

        # Overall bias score recommendation
        if report.overall_bias_score > 0.7:
            recommendations.append(
                f"CRITICAL: Overall bias score is {report.overall_bias_score:.2f} (HIGH). "
                "Strongly recommend comprehensive debiasing."
            )
        elif report.overall_bias_score > 0.4:
            recommendations.append(
                f"WARNING: Overall bias score is {report.overall_bias_score:.2f} (MODERATE). "
                "Recommend targeted debiasing interventions."
            )
        elif report.overall_bias_score > 0.2:
            recommendations.append(
                f"INFO: Overall bias score is {report.overall_bias_score:.2f} (LOW). "
                "Consider optional debiasing for high-risk decisions."
            )

        return recommendations

    def get_statistics(self) -> Dict:
        """Get statistics about bias detection history"""
        if not self.detection_history:
            return {
                "total_analyses": 0,
                "total_detections": 0,
                "average_bias_score": 0.0
            }

        total_detections = sum(r.total_detections for r in self.detection_history)
        avg_bias_score = sum(r.overall_bias_score for r in self.detection_history) / len(self.detection_history)

        return {
            "total_analyses": len(self.detection_history),
            "total_detections": total_detections,
            "average_bias_score": avg_bias_score,
            "most_common_biases": self._get_most_common_biases()
        }

    def _get_most_common_biases(self, top_n: int = 5) -> List[Tuple[BiasType, int]]:
        """Get the most commonly detected biases"""
        all_detections = []
        for report in self.detection_history:
            all_detections.extend(report.detections)

        bias_counts = Counter(d.bias_type for d in all_detections)
        return bias_counts.most_common(top_n)


# ========================================
# DEBIASING STRATEGIES
# ========================================

class DebiasingStrategy:
    """Base class for debiasing strategies"""

    @staticmethod
    def consider_the_opposite(constraint: Constraint) -> str:
        """
        Generate the opposite of a constraint to challenge assumptions.

        Returns:
            Suggested alternative formulation
        """
        # Extract key terms
        desc = constraint.description

        # Simple negation strategy
        opposites = {
            "must": "must not",
            "should": "should not",
            "required": "forbidden",
            "less than": "greater than",
            "greater than": "less than",
            "always": "never",
            "increase": "decrease",
            "maximize": "minimize",
        }

        new_desc = desc
        for pos, neg in opposites.items():
            new_desc = new_desc.replace(pos, f"[{neg}]")

        return f"Consider opposite: '{new_desc}'"

    @staticmethod
    def pre_mortem_analysis(
        constraints: List[Constraint],
        solution: Optional[str] = None
    ) -> List[str]:
        """
        Generate potential failure modes (pre-mortem analysis).

        Returns:
            List of potential failure scenarios
        """
        failure_modes = []

        # Generate failure modes based on constraint conflicts
        for i, c1 in enumerate(constraints):
            for c2 in constraints[i+1:]:
                # Check for potential conflicts
                if "less than" in c1.description and "greater than" in c2.description:
                    failure_modes.append(
                        f"Conflict between '{c1.id}' and '{c2.id}' could render solution infeasible"
                    )
                if "required" in c1.description and "forbidden" in c2.description:
                    failure_modes.append(
                        f"Contradiction between '{c1.id}' and '{c2.id}' could invalidate approach"
                    )

        return failure_modes

    @staticmethod
    def devils_advocate(constraint: Constraint) -> List[str]:
        """
        Generate devil's advocate challenges to a constraint.

        Returns:
            List of challenges
        """
        challenges = []

        # Challenge implicit assumptions
        challenges.append(f"Challenge: What if '{constraint.id}' is based on a false assumption?")

        # Challenge necessity
        challenges.append(f"Challenge: Is '{constraint.id}' truly necessary, or can we relax it?")

        # Challenge source
        challenges.append(
            f"Challenge: What evidence supports '{constraint.id}' beyond source '{constraint.source}'?"
        )

        return challenges

    @staticmethod
    def forced_reformulation(constraint: Constraint) -> List[str]:
        """
        Generate alternative reformulations of a constraint.

        Returns:
            List of alternative formulations
        """
        reformulations = []

        # Original
        reformulations.append(f"Original: {constraint.description}")

        # Positive frame
        pos_frame = constraint.description
        for neg_term in ["not", "never", "avoid", "prevent"]:
            pos_frame = pos_frame.replace(neg_term, "achieve")
        reformulations.append(f"Positive frame: {pos_frame}")

        # Negative frame
        neg_frame = constraint.description
        for pos_term in ["achieve", "gain", "obtain"]:
            neg_frame = neg_frame.replace(pos_term, "avoid")
        reformulations.append(f"Negative frame: {neg_frame}")

        # Quantitative frame (if applicable)
        if "less than" in constraint.description.lower():
            reformulations.append("Quantitative: Specify exact tolerance bounds")

        return reformulations


# ========================================
# DEMONSTRATION AND TESTING
# ========================================

if __name__ == "__main__":
    print("=" * 80)
    print("Φ₂ Metacognitive Debiasing System - Demonstration")
    print("=" * 80)

    # Create detector instance
    detector = CognitiveBiasDetector()
    print("\n[OK] Φ₂ detector initialized")

    # Create test constraints with various biases
    test_constraints = [
        Constraint(
            id="c1",
            type=ConstraintType.HARD,
            description="The system will certainly achieve 100% accuracy without any errors",
            formalization="accuracy = 1.0",
            source="user_prompt"
        ),
        Constraint(
            id="c2",
            type=ConstraintType.HARD,
            description="Clearly, this approach is obviously the best solution",
            formalization="best_solution = current_approach",
            source="expert_opinion"
        ),
        Constraint(
            id="c3",
            type=ConstraintType.SOFT,
            description="We must avoid failure at all costs due to our previous investment",
            formalization="failure = forbidden",
            source="existing_work"
        ),
        Constraint(
            id="c4",
            type=ConstraintType.PREFERENCE,
            description="The temperature precisely equals 100 degrees",
            formalization="T = 100",
            source="user_prompt"
        ),
        Constraint(
            id="c5",
            type=ConstraintType.HARD,
            description="In retrospect, the data clearly shows our initial assumptions were correct",
            formalization="assumptions_correct = true",
            source="post_hoc_analysis"
        ),
    ]

    print(f"\n[OK] Created {len(test_constraints)} test constraints with various biases")

    # Run analysis
    print("\n" + "=" * 80)
    print("Running Bias Analysis...")
    print("=" * 80)

    report = detector.analyze_constraints(test_constraints)

    # Display results
    print(f"\nTotal detections: {report.total_detections}")
    print(f"Overall bias score: {report.overall_bias_score:.2f}")

    print("\n" + "-" * 80)
    print("Detections by Type:")
    print("-" * 80)
    for bias_type, count in report.detections_by_type.items():
        print(f"  {bias_type.value}: {count}")

    print("\n" + "-" * 80)
    print("Detections by Severity:")
    print("-" * 80)
    for severity, count in report.detections_by_severity.items():
        print(f"  {severity.name}: {count}")

    print("\n" + "-" * 80)
    print("Top 5 Detections:")
    print("-" * 80)
    sorted_detections = sorted(
        report.detections,
        key=lambda d: (d.severity.value, d.confidence),
        reverse=True
    )
    for i, detection in enumerate(sorted_detections[:5], 1):
        print(f"\n{i}. {detection.bias_type.value} [{detection.severity.name}]")
        print(f"   Confidence: {detection.confidence:.2f}")
        print(f"   Description: {detection.description}")
        if detection.evidence:
            print(f"   Evidence: {detection.evidence}")
        if detection.suggestion:
            print(f"   Suggestion: {detection.suggestion}")

    print("\n" + "-" * 80)
    print("Recommendations:")
    print("-" * 80)
    for i, rec in enumerate(report.recommendations, 1):
        print(f"{i}. {rec}")

    # Demonstrate debiasing strategies
    print("\n" + "=" * 80)
    print("Debiasing Strategies Demonstration")
    print("=" * 80)

    print("\n1. Consider the Opposite:")
    print(DebiasingStrategy.consider_the_opposite(test_constraints[0]))

    print("\n2. Devil's Advocate:")
    for challenge in DebiasingStrategy.devils_advocate(test_constraints[1]):
        print(f"  - {challenge}")

    print("\n3. Pre-Mortem Analysis:")
    failure_modes = DebiasingStrategy.pre_mortem_analysis(test_constraints)
    for mode in failure_modes[:3]:
        print(f"  - {mode}")

    print("\n4. Forced Reformulation:")
    for reform in DebiasingStrategy.forced_reformulation(test_constraints[2]):
        print(f"  - {reform}")

    # Statistics
    print("\n" + "=" * 80)
    print("Statistics:")
    print("=" * 80)
    stats = detector.get_statistics()
    for key, value in stats.items():
        if isinstance(value, list):
            print(f"\n{key}:")
            for item in value:
                print(f"  - {item[0].value}: {item[1]}")
        else:
            print(f"  {key}: {value}")

    print("\n" + "=" * 80)
    print("[OK] Φ₂ demonstration complete")
    print("=" * 80)
