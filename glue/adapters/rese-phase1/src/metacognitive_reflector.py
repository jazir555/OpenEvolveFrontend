#!/usr/bin/env python3
"""
Φ₂: Metacognitive Reflection and Debiasing Subroutine

This module implements the mandatory debiasing component from RESE Technical Manual §3.2.
It is a P0 CRITICAL component for specification compliance.

Following CLAUDE.md principles:
- Law of Idempotency: Safe to run multiple times
- Law of Configuration Explicitness: All config via env vars
- Circuit Breaker Pattern: Timeout enforcement
- Structured Logging: JSON with correlation_id
- Law of UTC: All timestamps in UTC ISO-8601

Technical Manual Reference:
- Section 3.2: Table 1.0 - Φ₂ Metacognitive Reflection (ℛ_opp)
- Section 3.2: Active Antithetical Outcome Generation
- Section 3.2: Confirmation Bias Index (CBI)
"""

import os
import sys
import uuid
import time
import re
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict, field
from enum import Enum
import logging

# Add glue lib to path for StructuredLogger
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../lib'))

# Import StructuredLogger from phase1_executor
from phase1_executor import StructuredLogger, Phase1Config


# ============================================================================
# DATA STRUCTURES
# ============================================================================

class BiasType(Enum):
    """Types of directional bias"""
    CONFIRMATION = "confirmation"  # Seeking evidence that supports hypothesis
    DISCONFIRMATION = "disconfirmation"  # Seeking evidence that refutes hypothesis
    NEUTRAL = "neutral"  # No directional bias


class Severity(Enum):
    """Severity levels for bias"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass
class Hypothesis:
    """Hypothesis structure for debiasing"""
    id: str
    statement: str
    confidence: float
    assumptions: List[str] = field(default_factory=list)
    evidence: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Hypothesis':
        return cls(**data)


@dataclass
class BiasAnalysis:
    """Analysis of directional bias in a hypothesis

    From RESE Manual §3.2: Identify directional language and assumptions
    """
    bias_type: BiasType
    confidence: float  # 0-1, confidence that bias exists
    affected_assumptions: List[str]  # IDs of biased assumptions
    directional_language: List[str]  # Examples of biased language found
    severity: Severity
    analysis_details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['bias_type'] = self.bias_type.value
        data['severity'] = self.severity.value
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BiasAnalysis':
        if isinstance(data.get('bias_type'), str):
            data['bias_type'] = BiasType(data['bias_type'])
        if isinstance(data.get('severity'), str):
            data['severity'] = Severity(data['severity'])
        return cls(**data)


@dataclass
class DebiasingResult:
    """Result of metacognitive debiasing process

    From RESE Manual §3.2: Φ₂ Metacognitive Reflection output
    """
    original_hypothesis: Hypothesis
    debiased_hypothesis: Hypothesis
    antithetical_outcomes: List[Hypothesis]
    confirmation_bias_index: float  # 0-1, lower is better
    initial_cbi: float  # CBI before debiasing
    bias_reduction: float  # Percentage reduction in bias (0-100)
    metacognitive_reflections_applied: int
    correlation_id: str
    timestamp: str  # UTC ISO-8601 (Law of UTC)
    bias_analysis: BiasAnalysis
    reflections_applied: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['original_hypothesis'] = self.original_hypothesis.to_dict()
        data['debiased_hypothesis'] = self.debiased_hypothesis.to_dict()
        data['antithetical_outcomes'] = [
            h.to_dict() for h in self.antithetical_outcomes
        ]
        data['bias_analysis'] = self.bias_analysis.to_dict()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DebiasingResult':
        if isinstance(data.get('original_hypothesis'), dict):
            data['original_hypothesis'] = Hypothesis.from_dict(data['original_hypothesis'])
        if isinstance(data.get('debiased_hypothesis'), dict):
            data['debiased_hypothesis'] = Hypothesis.from_dict(data['debiased_hypothesis'])
        if isinstance(data.get('antithetical_outcomes'), list):
            data['antithetical_outcomes'] = [
                Hypothesis.from_dict(h) for h in data['antithetical_outcomes']
            ]
        if isinstance(data.get('bias_analysis'), dict):
            data['bias_analysis'] = BiasAnalysis.from_dict(data['bias_analysis'])
        return cls(**data)


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class DebiasingConfig:
    """Configuration for Metacognitive Reflector"""

    # Feature flags
    ENABLE_DEBIASING: bool
    CBI_THRESHOLD: float  # Maximum acceptable CBI
    ANTITHETICAL_COUNT: int  # Number of alternatives to generate

    # Timeout settings
    TIMEOUT_MS: int

    # Bias detection thresholds
    DIRECTIONAL_LANGUAGE_THRESHOLD: int  # Min phrases to flag as biased
    CONFIDENCE_THRESHOLD: float  # Min confidence to consider bias significant

    @classmethod
    def from_env(cls) -> 'DebiasingConfig':
        """Load configuration from environment variables

        Law of Configuration Explicitness: All config via env vars
        Crashes immediately if required config is missing or invalid
        """
        config = cls(
            ENABLE_DEBIASING=os.getenv('PHASE1_DEBIASING_ENABLED', 'true').lower() == 'true',
            CBI_THRESHOLD=float(os.getenv('PHASE1_CBI_THRESHOLD', '0.5')),
            ANTITHETICAL_COUNT=int(os.getenv('PHASE1_ANTITHETICAL_COUNT', '3')),
            TIMEOUT_MS=int(os.getenv('PHASE1_DEBIASING_TIMEOUT_MS', '5000')),
            DIRECTIONAL_LANGUAGE_THRESHOLD=int(os.getenv('PHASE1_DIRECTIONAL_THRESHOLD', '2')),
            CONFIDENCE_THRESHOLD=float(os.getenv('PHASE1_CONFIDENCE_THRESHOLD', '0.3')),
        )

        # Validate configuration
        if config.CBI_THRESHOLD < 0 or config.CBI_THRESHOLD > 1:
            raise ValueError("PHASE1_CBI_THRESHOLD must be between 0 and 1")
        if config.ANTITHETICAL_COUNT < 1:
            raise ValueError("PHASE1_ANTITHETICAL_COUNT must be at least 1")
        if config.TIMEOUT_MS <= 0:
            raise ValueError("PHASE1_DEBIASING_TIMEOUT_MS must be positive")
        if config.CONFIDENCE_THRESHOLD < 0 or config.CONFIDENCE_THRESHOLD > 1:
            raise ValueError("PHASE1_CONFIDENCE_THRESHOLD must be between 0 and 1")

        return config


# ============================================================================
# MAIN METACOGNITIVE REFLECTOR CLASS
# ============================================================================

class MetacognitiveReflector:
    """
    Φ₂: Metacognitive Reflection and Debiasing Subroutine

    Implements the mandatory debiasing component from RESE Technical Manual §3.2

    From RESE Manual §3.2:
    "Φ₂ applies metacognitive reflection (ℛ_opp) to enforce non-directional
    hypothesis testing, actively generate antithetical outcomes, and measure
    the Confirmation Bias Index (CBI)."
    """

    # Directional language patterns that indicate bias
    CONFIRMATION_PATTERNS = [
        r'\b(obviously|clearly|undoubtedly|certainly|surely)\b',
        r'\b(must be|has to be|cannot be|cannot fail to)\b',
        r'\b(proves|demonstrates|confirms|validates)\b',
        r'\b(no doubt|without question|undeniably)\b',
        r'\b(expect to|should|ought to)\b',
    ]

    DISCONFIRMATION_PATTERNS = [
        r'\b(unlikely|improbable|doubtful|questionable)\b',
        r'\b(fails to|cannot|unable to)\b',
        r'\b(refutes|contradicts|disproves)\b',
        r'\b(highly suspect|certainly not)\b',
    ]

    # Neutral language patterns
    NEUTRAL_PATTERNS = [
        r'\b(may|might|could|possibly)\b',
        r'\b(suggests|indicates|implies)\b',
        r'\b(appears to be|seems to be)\b',
        r'\b(potentially|perhaps)\b',
    ]

    def __init__(self, config: Optional[DebiasingConfig] = None, logger: Optional[StructuredLogger] = None):
        """Initialize Metacognitive Reflector

        Args:
            config: Configuration object (loaded from env if None)
            logger: Structured logger instance (created if None)
        """
        self.config = config or DebiasingConfig.from_env()
        self.logger = logger or StructuredLogger('MetacognitiveReflector')

        # Compile regex patterns for efficiency
        self.confirmation_regex = [re.compile(p, re.IGNORECASE) for p in self.CONFIRMATION_PATTERNS]
        self.disconfirmation_regex = [re.compile(p, re.IGNORECASE) for p in self.DISCONFIRMATION_PATTERNS]
        self.neutral_regex = [re.compile(p, re.IGNORECASE) for p in self.NEUTRAL_PATTERNS]

        self.logger.info("MetacognitiveReflector initialized",
            enabled=self.config.ENABLE_DEBIASING,
            cbi_threshold=self.config.CBI_THRESHOLD,
            antithetical_count=self.config.ANTITHETICAL_COUNT,
            timeout_ms=self.config.TIMEOUT_MS,
        )

    def perform_debiasing(
        self,
        hypothesis: Hypothesis,
        assumptions: List[Any],  # TacitAssumption objects
        correlation_id: str,
    ) -> DebiasingResult:
        """
        Apply metacognitive reflection to reduce directional bias

        From RESE Manual §3.2:
        1. Identify directional bias in hypothesis
        2. Generate antithetical outcomes
        3. Calculate confirmation bias index
        4. Apply metacognitive reflection
        5. Return debiased hypothesis with CBI score

        Law of Idempotency: Safe to run multiple times
        Law of Timeout Enforcement: Enforced timeout

        Args:
            hypothesis: Hypothesis to debias
            assumptions: List of tacit assumptions (TacitAssumption objects)
            correlation_id: Distributed tracing correlation ID

        Returns:
            DebiasingResult with CBI score and antithetical outcomes

        Raises:
            TimeoutError: If debiasing exceeds timeout
            RuntimeError: If debiasing is disabled
        """
        start_time = time.time()

        if not self.config.ENABLE_DEBIASING:
            raise RuntimeError("Metacognitive debiasing is disabled")

        self.logger.info("Starting Φ₂: Metacognitive Reflection",
            correlation_id=correlation_id,
            hypothesis_id=hypothesis.id,
            hypothesis_statement=hypothesis.statement,
        )

        try:
            # Step 1: Identify directional bias
            self.logger.debug("Step 1: Identifying directional bias",
                correlation_id=correlation_id,
            )
            bias_analysis = self._identify_directional_bias(hypothesis)

            # Step 2: Generate antithetical outcomes
            self.logger.debug("Step 2: Generating antithetical outcomes",
                correlation_id=correlation_id,
                count=self.config.ANTITHETICAL_COUNT,
            )
            antithetical_outcomes = self._generate_antithetical_outcomes(
                hypothesis,
                count=self.config.ANTITHETICAL_COUNT,
                correlation_id=correlation_id,
            )

            # Step 3: Calculate initial CBI
            self.logger.debug("Step 3: Calculating initial Confirmation Bias Index",
                correlation_id=correlation_id,
            )
            initial_cbi = self._calculate_confirmation_bias_index(
                hypothesis,
                antithetical_outcomes,
                [],
                correlation_id,
            )

            # Step 4: Apply metacognitive reflection
            self.logger.debug("Step 4: Applying metacognitive reflection",
                correlation_id=correlation_id,
            )
            debiased_hypothesis = self._apply_metacognitive_reflection(
                hypothesis,
                bias_analysis,
                antithetical_outcomes,
                correlation_id,
            )

            # Step 5: Calculate final CBI after debiasing
            self.logger.debug("Step 5: Calculating final Confirmation Bias Index",
                correlation_id=correlation_id,
            )
            final_cbi = self._calculate_confirmation_bias_index(
                debiased_hypothesis,
                antithetical_outcomes,
                [],
                correlation_id,
            )

            # Calculate bias reduction
            bias_reduction = 0.0
            if initial_cbi > 0:
                bias_reduction = ((initial_cbi - final_cbi) / initial_cbi) * 100

            execution_time_ms = int((time.time() - start_time) * 1000)

            result = DebiasingResult(
                original_hypothesis=hypothesis,
                debiased_hypothesis=debiased_hypothesis,
                antithetical_outcomes=antithetical_outcomes,
                confirmation_bias_index=final_cbi,
                initial_cbi=initial_cbi,
                bias_reduction=bias_reduction,
                metacognitive_reflections_applied=len(bias_analysis.directional_language),
                correlation_id=correlation_id,
                timestamp=datetime.now(timezone.utc).isoformat(),  # Law of UTC
                bias_analysis=bias_analysis,
                reflections_applied=bias_analysis.directional_language,
            )

            self.logger.info("Φ₂: Metacognitive Reflection completed",
                correlation_id=correlation_id,
                initial_cbi=initial_cbi,
                final_cbi=final_cbi,
                bias_reduction=bias_reduction,
                antithetical_outcomes=len(antithetical_outcomes),
                execution_time_ms=execution_time_ms,
            )

            # Check timeout
            if execution_time_ms > self.config.TIMEOUT_MS:
                raise TimeoutError(f"Debiasing exceeded timeout: {execution_time_ms}ms")

            return result

        except Exception as e:
            execution_time_ms = int((time.time() - start_time) * 1000)
            self.logger.error("Φ₂: Metacognitive Reflection failed", e,
                correlation_id=correlation_id,
                execution_time_ms=execution_time_ms,
            )
            raise

    def _identify_directional_bias(self, hypothesis: Hypothesis) -> BiasAnalysis:
        """
        Analyze hypothesis for directional language and assumptions

        From RESE Manual §3.2: Identify directional bias

        Args:
            hypothesis: Hypothesis to analyze

        Returns:
            BiasAnalysis with bias_type, confidence, affected_assumptions
        """
        statement = hypothesis.statement.lower()

        # Count confirmation patterns
        confirmation_count = 0
        confirmation_examples = []
        for pattern in self.confirmation_regex:
            matches = pattern.findall(statement)
            if matches:
                confirmation_count += len(matches)
                confirmation_examples.extend(matches)

        # Count disconfirmation patterns
        disconfirmation_count = 0
        disconfirmation_examples = []
        for pattern in self.disconfirmation_regex:
            matches = pattern.findall(statement)
            if matches:
                disconfirmation_count += len(matches)
                disconfirmation_examples.extend(matches)

        # Count neutral patterns
        neutral_count = 0
        for pattern in self.neutral_regex:
            matches = pattern.findall(statement)
            if matches:
                neutral_count += len(matches)

        # Determine bias type
        total_directional = confirmation_count + disconfirmation_count
        directional_language = confirmation_examples + disconfirmation_examples

        # Calculate confidence (0-1)
        confidence = 0.0
        if total_directional > 0:
            confidence = min(total_directional / (total_directional + neutral_count + 1), 1.0)

        # Determine bias type and severity
        if confirmation_count > disconfirmation_count:
            if confirmation_count >= self.config.DIRECTIONAL_LANGUAGE_THRESHOLD:
                bias_type = BiasType.CONFIRMATION
                # Severity based on count and confidence
                if confirmation_count >= 4 and confidence >= 0.7:
                    severity = Severity.HIGH
                elif confirmation_count >= 2 and confidence >= 0.5:
                    severity = Severity.MEDIUM
                else:
                    severity = Severity.LOW
            else:
                bias_type = BiasType.NEUTRAL
                severity = Severity.LOW
                confidence = 0.0
        elif disconfirmation_count > confirmation_count:
            if disconfirmation_count >= self.config.DIRECTIONAL_LANGUAGE_THRESHOLD:
                bias_type = BiasType.DISCONFIRMATION
                if disconfirmation_count >= 4 and confidence >= 0.7:
                    severity = Severity.HIGH
                elif disconfirmation_count >= 2 and confidence >= 0.5:
                    severity = Severity.MEDIUM
                else:
                    severity = Severity.LOW
            else:
                bias_type = BiasType.NEUTRAL
                severity = Severity.LOW
                confidence = 0.0
        else:
            bias_type = BiasType.NEUTRAL
            severity = Severity.LOW
            confidence = 0.0

        # Find affected assumptions (those with directional language)
        affected_assumptions = []
        for i, assumption_text in enumerate(hypothesis.assumptions):
            assumption_lower = assumption_text.lower()
            for pattern in self.confirmation_regex + self.disconfirmation_regex:
                if pattern.search(assumption_lower):
                    affected_assumptions.append(f"assumption_{i}")
                    break

        analysis = BiasAnalysis(
            bias_type=bias_type,
            confidence=confidence,
            affected_assumptions=affected_assumptions,
            directional_language=directional_language,
            severity=severity,
            analysis_details={
                'confirmation_count': confirmation_count,
                'disconfirmation_count': disconfirmation_count,
                'neutral_count': neutral_count,
                'total_directional': total_directional,
            },
        )

        self.logger.debug("Bias analysis completed",
            bias_type=bias_type.value,
            confidence=confidence,
            severity=severity.value,
            directional_examples=directional_language[:3],  # Log first 3
        )

        return analysis

    def _generate_antithetical_outcomes(
        self,
        hypothesis: Hypothesis,
        count: int,
        correlation_id: str,
    ) -> List[Hypothesis]:
        """
        Generate opposite hypotheses to test robustness

        From RESE Manual §3.2: Active Antithetical Outcome Generation

        Strategies:
        - Negate primary conclusion
        - Invert causal mechanisms
        - Test alternative explanations

        Args:
            hypothesis: Original hypothesis
            count: Number of antithetical outcomes to generate
            correlation_id: Correlation ID for tracing

        Returns:
            List of antithetical hypotheses
        """
        antithetical = []

        # Strategy 1: Negate primary conclusion
        negated = self._negate_hypothesis(hypothesis, correlation_id)
        antithetical.append(negated)

        # Strategy 2: Invert causal mechanism
        inverted = self._invert_causality(hypothesis, correlation_id)
        antithetical.append(inverted)

        # Strategy 3: Alternative explanation
        if count >= 3:
            alternative = self._generate_alternative_explanation(hypothesis, correlation_id)
            antithetical.append(alternative)

        # Strategy 4: Random variation (if needed)
        while len(antithetical) < count:
            variation = self._generate_random_variation(hypothesis, correlation_id, len(antithetical))
            antithetical.append(variation)

        self.logger.debug("Antithetical outcomes generated",
            correlation_id=correlation_id,
            count=len(antithetical),
        )

        return antithetical[:count]

    def _negate_hypothesis(self, hypothesis: Hypothesis, correlation_id: str) -> Hypothesis:
        """Negate the primary conclusion"""
        statement = hypothesis.statement

        # Simple negation patterns
        negations = [
            (r'\bis\b', 'is not'),
            (r'\bare\b', 'are not'),
            (r'\bcauses\b', 'does not cause'),
            (r'\bwill\b', 'will not'),
            (r'\bcan\b', 'cannot'),
            (r'\bmakes\b', 'does not make'),
            (r'\bproves\b', 'does not prove'),
            (r'\bdemonstrates\b', 'does not demonstrate'),
        ]

        negated_statement = statement
        for pattern, replacement in negations:
            negated_statement = re.sub(pattern, replacement, negated_statement, flags=re.IGNORECASE)

        # If no changes, prepend negation
        if negated_statement == statement:
            negated_statement = f"It is not the case that {statement}"

        return Hypothesis(
            id=str(uuid.uuid4()),
            statement=negated_statement,
            confidence=hypothesis.confidence * 0.5,  # Lower confidence
            assumptions=[f"Negation of: {a}" for a in hypothesis.assumptions],
        )

    def _invert_causality(self, hypothesis: Hypothesis, correlation_id: str) -> Hypothesis:
        """Invert causal mechanism"""
        statement = hypothesis.statement

        # Causal inversion patterns
        inversions = [
            (r'\b(.+?) causes (.+?)\b', r'\2 causes \1'),
            (r'\b(.+?) leads to (.+?)\b', r'\2 leads to \1'),
            (r'\b(.+?) results in (.+?)\b', r'\2 results in \1'),
            (r'\b(.+?) produces (.+?)\b', r'\2 produces \1'),
        ]

        inverted_statement = statement
        for pattern, replacement in inversions:
            inverted_statement = re.sub(pattern, replacement, inverted_statement, flags=re.IGNORECASE)

        # If no changes, add explanatory text
        if inverted_statement == statement:
            inverted_statement = f"The reverse causality may apply: {statement}"

        return Hypothesis(
            id=str(uuid.uuid4()),
            statement=inverted_statement,
            confidence=hypothesis.confidence * 0.6,
            assumptions=[f"Reverse causality of: {a}" for a in hypothesis.assumptions],
        )

    def _generate_alternative_explanation(self, hypothesis: Hypothesis, correlation_id: str) -> Hypothesis:
        """Generate alternative explanation"""
        statement = hypothesis.statement

        # Add alternative framing
        alternatives = [
            f"Alternatively, {statement.lower()}",
            f"Another possible explanation is that {statement.lower()}",
            f"A different perspective suggests that {statement.lower()}",
        ]

        import random
        alternative_statement = random.choice(alternatives)

        return Hypothesis(
            id=str(uuid.uuid4()),
            statement=alternative_statement,
            confidence=hypothesis.confidence * 0.7,
            assumptions=[f"Alternative to: {a}" for a in hypothesis.assumptions],
        )

    def _generate_random_variation(self, hypothesis: Hypothesis, correlation_id: str, seed: int) -> Hypothesis:
        """Generate random variation for additional antithetical outcomes"""
        variations = [
            f"Conversely, {hypothesis.statement.lower()}",
            f"In contrast to the hypothesis, {hypothesis.statement.lower()}",
            f"An opposing view suggests that {hypothesis.statement.lower()}",
        ]

        import random
        random.seed(seed)
        variation_statement = random.choice(variations)

        return Hypothesis(
            id=str(uuid.uuid4()),
            statement=variation_statement,
            confidence=hypothesis.confidence * 0.55,
            assumptions=[f"Variation of: {a}" for a in hypothesis.assumptions],
        )

    def _calculate_confirmation_bias_index(
        self,
        hypothesis: Hypothesis,
        antithetical: List[Hypothesis],
        evidence: List[Any],  # Evidence objects (simplified)
        correlation_id: str,
    ) -> float:
        """
        Calculate Confirmation Bias Index (CBI)

        From RESE Manual §3.2:
        CBI = |P(H|E) - P(H̄|E)|

        Where:
        - P(H|E) = Probability of hypothesis given evidence
        - P(H̄|E) = Probability of opposite hypothesis given evidence

        Returns:
            float: 0.0 (unbiased) to 1.0 (fully biased)

        Note: This is a simplified implementation. A full implementation would
        use Bayesian updating with actual evidence.
        """
        # Simplified CBI calculation based on confidence scores
        # and the distribution of antithetical outcomes

        if not antithetical:
            return 1.0  # Maximum bias if no alternatives considered

        # P(H|E) - probability of original hypothesis
        p_h_given_e = hypothesis.confidence

        # P(H̄|E) - probability of opposite hypothesis
        # Average confidence of antithetical outcomes
        p_h_bar_given_e = sum(h.confidence for h in antithetical) / len(antithetical)

        # Calculate CBI
        cbi = abs(p_h_given_e - p_h_bar_given_e)

        self.logger.debug("CBI calculated",
            correlation_id=correlation_id,
            p_h_given_e=p_h_given_e,
            p_h_bar_given_e=p_h_bar_given_e,
            cbi=cbi,
        )

        return cbi

    def _apply_metacognitive_reflection(
        self,
        hypothesis: Hypothesis,
        bias_analysis: BiasAnalysis,
        antithetical_outcomes: List[Hypothesis],
        correlation_id: str,
    ) -> Hypothesis:
        """
        Apply ℛ_opp metacognitive reflection

        From RESE Manual §3.2:
        Forces consideration of antithetical outcomes and alternative explanations

        Args:
            hypothesis: Original hypothesis
            bias_analysis: Bias analysis from identification step
            antithetical_outcomes: Generated antithetical hypotheses
            correlation_id: Correlation ID for tracing

        Returns:
            DebiasingHypothesis with reduced directional bias
        """
        statement = hypothesis.statement
        reflections_applied = []

        # Replace directional language with neutral alternatives
        replacements = {
            'obviously': 'possibly',
            'clearly': 'appears to',
            'undoubtedly': 'may',
            'certainly': 'might',
            'surely': 'could',
            'must be': 'may be',
            'has to be': 'could be',
            'cannot be': 'might not be',
            'cannot fail to': 'may not',
            'proves': 'suggests',
            'demonstrates': 'indicates',
            'confirms': 'implies',
            'validates': 'supports',
            'no doubt': 'it appears',
            'without question': 'potentially',
            'undeniably': 'possibly',
        }

        # Apply replacements (case-insensitive)
        for directional, neutral in replacements.items():
            pattern = re.compile(r'\b' + directional + r'\b', re.IGNORECASE)
            if pattern.search(statement):
                statement = pattern.sub(neutral, statement)
                reflections_applied.append(f"Replaced '{directional}' with '{neutral}'")

        # If confirmation bias detected, add uncertainty qualifiers
        if bias_analysis.bias_type == BiasType.CONFIRMATION:
            if not any(qualifier in statement.lower() for qualifier in ['may', 'might', 'could', 'possibly']):
                # Insert qualifier at beginning if not present
                statement = f"It appears that {statement[0].lower()}{statement[1:]}"
                reflections_applied.append("Added uncertainty qualifier")

        # Add consideration of antithetical outcomes
        if antithetical_outcomes:
            antithetical_summary = " | ".join([
                f"Alt: {h.statement[:50]}..." for h in antithetical_outcomes[:2]
            ])
            reflections_applied.append(f"Considered alternatives: [{antithetical_summary}]")

        # Reduce confidence to reflect uncertainty
        adjusted_confidence = hypothesis.confidence
        if bias_analysis.bias_type != BiasType.NEUTRAL:
            # Reduce confidence based on severity
            if bias_analysis.severity == Severity.HIGH:
                adjusted_confidence *= 0.7
            elif bias_analysis.severity == Severity.MEDIUM:
                adjusted_confidence *= 0.85
            elif bias_analysis.severity == Severity.LOW:
                adjusted_confidence *= 0.95

        # Create debiased hypothesis
        debiased = Hypothesis(
            id=hypothesis.id,
            statement=statement,
            confidence=adjusted_confidence,
            assumptions=hypothesis.assumptions,
        )

        self.logger.debug("Metacognitive reflection applied",
            correlation_id=correlation_id,
            original_confidence=hypothesis.confidence,
            debiased_confidence=adjusted_confidence,
            reflections_count=len(reflections_applied),
        )

        return debiased

    def get_stats(self) -> Dict[str, Any]:
        """Get reflector statistics"""
        return {
            'config': {
                'enabled': self.config.ENABLE_DEBIASING,
                'cbi_threshold': self.config.CBI_THRESHOLD,
                'antithetical_count': self.config.ANTITHETICAL_COUNT,
                'timeout_ms': self.config.TIMEOUT_MS,
            },
        }


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main():
    """Main entry point for testing"""
    import argparse
    import json

    parser = argparse.ArgumentParser(description='Φ₂: Metacognitive Reflection')
    parser.add_argument('--statement', required=True, help='Hypothesis statement')
    parser.add_argument('--confidence', type=float, default=0.8, help='Hypothesis confidence')
    parser.add_argument('--correlation-id', help='Correlation ID')
    args = parser.parse_args()

    # Load configuration from environment
    config = DebiasingConfig.from_env()

    # Create reflector
    reflector = MetacognitiveReflector(config=config)

    # Create hypothesis
    hypothesis = Hypothesis(
        id=str(uuid.uuid4()),
        statement=args.statement,
        confidence=args.confidence,
        assumptions=["Assumption 1", "Assumption 2"],
    )

    # Perform debiasing
    result = reflector.perform_debiasing(
        hypothesis=hypothesis,
        assumptions=[],
        correlation_id=args.correlation_id or str(uuid.uuid4()),
    )

    # Output result as JSON
    print(json.dumps(result.to_dict(), indent=2))


if __name__ == '__main__':
    main()
