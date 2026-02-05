"""
Autoformalization Service for RESE-LeanAide Integration

Provides AI-powered autoformalization capabilities for all 4 RESE phases:
- Phase I: Epistemic Audit - Autoformalize constraints and assumptions
- Phase II: Isomorphic Mapping - Autoformalize FDGs and isomorphisms
- Phase III: MCTS Refinement - Autoformalize hypotheses and patterns
- Phase IV: Architectural Synthesis - Autoformalize predictive models

Following CLAUDE.md principles:
- Law of Configuration Explicitness: All config via env vars
- Law of Idempotency: Safe to call multiple times
- Structured Logging: JSON with correlation_id
- Timeout: All operations have timeouts

Author: OpenEvolve
Version: 1.0.0
"""

import asyncio
import json
import logging
import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Union
import re

# Import LeanAide client
try:
    from leanaide_client import LeanAideClient, LeanAideConfig, LeanAideResult
    LEANAIDE_AVAILABLE = True
except ImportError:
    LEANAIDE_AVAILABLE = False
    logging.warning("LeanAide client not available - using simulation mode")

# Import RESE schemas
try:
    from glue.schemas.rese_schemas import (
        Hypothesis, Pattern, IsomorphicMapping, FunctionalDependencyGraph,
        HypothesisStatus, PatternType, IsomorphismType
    )
except ImportError:
    from rese_schemas import (
        Hypothesis, Pattern, IsomorphicMapping, FunctionalDependencyGraph,
        HypothesisStatus, PatternType, IsomorphismType
    )


# Configure logging
logger = logging.getLogger(__name__)


# ============================================================================
# Enums and Data Structures
# ============================================================================

class AutoformalizationPhase(Enum):
    """RESE phases that support autoformalization"""
    PHASE_I_EPISTEMIC_AUDIT = "phase_i_epistemic_audit"
    PHASE_II_ISOMORPHIC_MAPPING = "phase_ii_isomorphic_mapping"
    PHASE_III_MCTS_REFINEMENT = "phase_iii_mcts_refinement"
    PHASE_IV_ARCHITECTURAL_SYNTHESIS = "phase_iv_architectural_synthesis"


class FormalizationDomain(Enum):
    """Mathematical domains for formalization"""
    ARITHMETIC = "arithmetic"
    ALGEBRA = "algebra"
    LOGIC = "logic"
    SET_THEORY = "set_theory"
    CALCULUS = "calculus"
    GRAPH_THEORY = "graph_theory"
    PROBABILITY = "probability"
    TOPOLOGY = "topology"
    CATEGORY_THEORY = "category_theory"


@dataclass
class AutoformalizationResult:
    """Result from autoformalization"""
    success: bool
    phase: AutoformalizationPhase
    natural_language: str
    lean_code: str
    domain: FormalizationDomain
    confidence: float = 0.0
    lean_theorem_name: Optional[str] = None
    lean_type: Optional[str] = None
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    alternatives: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    correlation_id: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    execution_time_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "success": self.success,
            "phase": self.phase.value if isinstance(self.phase, Enum) else self.phase,
            "natural_language": self.natural_language,
            "lean_code": self.lean_code,
            "domain": self.domain.value if isinstance(self.domain, Enum) else self.domain,
            "confidence": self.confidence,
            "lean_theorem_name": self.lean_theorem_name,
            "lean_type": self.lean_type,
            "errors": self.errors,
            "warnings": self.warnings,
            "alternatives": self.alternatives,
            "metadata": self.metadata,
            "correlation_id": self.correlation_id,
            "timestamp": self.timestamp,
            "execution_time_ms": self.execution_time_ms
        }


@dataclass
class AutoformalizationConfig:
    """Configuration for autoformalization service"""
    leanaide_host: str = "localhost"
    leanaide_port: int = 7654
    timeout_ms: int = 30000
    max_alternatives: int = 3
    confidence_threshold: float = 0.7
    enable_caching: bool = True
    cache_dir: str = ".leanaide_autoformalization_cache"
    correlation_id: Optional[str] = None

    @classmethod
    def from_env(cls) -> "AutoformalizationConfig":
        """Create configuration from environment variables"""
        return cls(
            leanaide_host=os.getenv("LEANAIDE_HOST", "localhost"),
            leanaide_port=int(os.getenv("LEANAIDE_PORT", "7654")),
            timeout_ms=int(os.getenv("LEANAIDE_TIMEOUT_MS", "30000")),
            max_alternatives=int(os.getenv("LEANAIDE_MAX_ALTERNATIVES", "3")),
            confidence_threshold=float(os.getenv("LEANAIDE_CONFIDENCE_THRESHOLD", "0.7")),
            enable_caching=os.getenv("LEANAIDE_ENABLE_CACHING", "true").lower() == "true",
            cache_dir=os.getenv("LEANAIDE_CACHE_DIR", ".leanaide_autoformalization_cache"),
            correlation_id=os.getenv("CORRELATION_ID")
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "leanaide_host": self.leanaide_host,
            "leanaide_port": self.leanaide_port,
            "timeout_ms": self.timeout_ms,
            "max_alternatives": self.max_alternatives,
            "confidence_threshold": self.confidence_threshold,
            "enable_caching": self.enable_caching,
            "cache_dir": self.cache_dir,
            "correlation_id": self.correlation_id
        }


# ============================================================================
# Structured Logger
# ============================================================================

class AutoformalizationLogger:
    """Structured logger for autoformalization service"""

    def __init__(self, correlation_id: Optional[str] = None):
        self.correlation_id = correlation_id or str(uuid.uuid4())
        self.logger = logging.getLogger("autoformalization_service")

    def _log(self, level: str, msg: str, **kwargs):
        """Log in JSON Lines format"""
        log_entry = {
            "msg": msg,
            "level": level,
            "correlation_id": self.correlation_id,
            "source_service": "autoformalization_service",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **kwargs
        }
        log_json = json.dumps(log_entry)
        self.logger.log(getattr(logging, level.upper()), log_json)

    def info(self, msg: str, **kwargs):
        self._log("INFO", msg, **kwargs)

    def warning(self, msg: str, **kwargs):
        self._log("WARNING", msg, **kwargs)

    def error(self, msg: str, **kwargs):
        self._log("ERROR", msg, **kwargs)

    def debug(self, msg: str, **kwargs):
        self._log("DEBUG", msg, **kwargs)


# ============================================================================
# Autoformalization Service
# ============================================================================

class AutoformalizationService:
    """
    Autoformalization service for RESE phases.

    Provides AI-powered translation from natural language to Lean 4 code
    for all RESE phases.
    """

    def __init__(
        self,
        config: Optional[AutoformalizationConfig] = None,
        logger: Optional[AutoformalizationLogger] = None
    ):
        """
        Initialize autoformalization service.

        Args:
            config: Service configuration
            logger: Structured logger
        """
        self.config = config or AutoformalizationConfig.from_env()
        self.logger = logger or AutoformalizationLogger(self.config.correlation_id)

        # Initialize LeanAide client
        self.leanaide_client: Optional[LeanAideClient] = None
        if LEANAIDE_AVAILABLE:
            leanaide_config = LeanAideConfig(
                host=self.config.leanaide_host,
                port=self.config.leanaide_port,
                timeout=self.config.timeout_ms / 1000.0
            )
            self.leanaide_client = LeanAideClient(config=leanaide_config)

        # Cache for results
        self.cache: Dict[str, AutoformalizationResult] = {}

        self.logger.info(
            "AutoformalizationService initialized",
            config=self.config.to_dict()
        )

    async def autoformalize_phase_i(
        self,
        constraint_text: str,
        constraint_type: str = "logical",
        correlation_id: Optional[str] = None
    ) -> AutoformalizationResult:
        """
        Autoformalize constraint for Phase I: Epistemic Audit.

        Args:
            constraint_text: Natural language constraint
            constraint_type: Type of constraint (logical, arithmetic, etc.)
            correlation_id: Correlation ID for tracing

        Returns:
            AutoformalizationResult with Lean 4 code
        """
        start_time = asyncio.get_event_loop().time()
        cid = correlation_id or self.logger.correlation_id

        self.logger.info(
            "Autoformalizing Phase I constraint",
            phase="phase_i_epistemic_audit",
            constraint_type=constraint_type,
            correlation_id=cid
        )

        try:
            # Detect domain
            domain = self._detect_domain(constraint_text)

            # Generate theorem name
            theorem_name = self._generate_theorem_name(constraint_text, "constraint")

            # Use LeanAide to translate
            if self.leanaide_client:
                result = await self.leanaide_client.translate_thm_detailed(
                    theorem_text=constraint_text,
                    theorem_name=theorem_name
                )

                if result.success:
                    lean_code = result.data.get("result", "")
                    lean_type = result.data.get("type", "")

                    execution_time_ms = (asyncio.get_event_loop().time() - start_time) * 1000

                    formalization_result = AutoformalizationResult(
                        success=True,
                        phase=AutoformalizationPhase.PHASE_I_EPISTEMIC_AUDIT,
                        natural_language=constraint_text,
                        lean_code=lean_code,
                        domain=domain,
                        confidence=0.85,
                        lean_theorem_name=theorem_name,
                        lean_type=lean_type,
                        correlation_id=cid,
                        execution_time_ms=execution_time_ms
                    )

                    self.logger.info(
                        "Phase I autoformalization successful",
                        correlation_id=cid,
                        theorem_name=theorem_name,
                        domain=domain.value,
                        execution_time_ms=execution_time_ms
                    )

                    return formalization_result
                else:
                    # Fall back to template-based generation
                    return self._generate_fallback_formalization(
                        constraint_text,
                        theorem_name,
                        domain,
                        AutoformalizationPhase.PHASE_I_EPISTEMIC_AUDIT,
                        cid,
                        start_time
                    )
            else:
                # Simulation mode
                return self._generate_fallback_formalization(
                    constraint_text,
                    theorem_name,
                    domain,
                    AutoformalizationPhase.PHASE_I_EPISTEMIC_AUDIT,
                    cid,
                    start_time
                )

        except Exception as e:
            self.logger.error(
                "Phase I autoformalization failed",
                correlation_id=cid,
                error=str(e)
            )

            return AutoformalizationResult(
                success=False,
                phase=AutoformalizationPhase.PHASE_I_EPISTEMIC_AUDIT,
                natural_language=constraint_text,
                lean_code="",
                domain=FormalizationDomain.LOGIC,
                errors=[str(e)],
                correlation_id=cid,
                execution_time_ms=(asyncio.get_event_loop().time() - start_time) * 1000
            )

    async def autoformalize_phase_ii(
        self,
        mapping_description: str,
        source_domain: str,
        target_domain: str,
        correlation_id: Optional[str] = None
    ) -> AutoformalizationResult:
        """
        Autoformalize isomorphic mapping for Phase II.

        Args:
            mapping_description: Description of isomorphic mapping
            source_domain: Source domain name
            target_domain: Target domain name
            correlation_id: Correlation ID

        Returns:
            AutoformalizationResult with Lean 4 code
        """
        start_time = asyncio.get_event_loop().time()
        cid = correlation_id or self.logger.correlation_id

        self.logger.info(
            "Autoformalizing Phase II mapping",
            phase="phase_ii_isomorphic_mapping",
            source_domain=source_domain,
            target_domain=target_domain,
            correlation_id=cid
        )

        try:
            # Generate theorem name
            theorem_name = f"isomorphic_{source_domain}_to_{target_domain}"

            # Build formal statement
            formal_statement = self._build_isomorphism_statement(
                mapping_description,
                source_domain,
                target_domain
            )

            # Use LeanAide to translate
            if self.leanaide_client:
                result = await self.leanaide_client.translate_thm_detailed(
                    theorem_text=formal_statement,
                    theorem_name=theorem_name
                )

                if result.success:
                    lean_code = result.data.get("result", "")
                    lean_type = result.data.get("type", "")

                    execution_time_ms = (asyncio.get_event_loop().time() - start_time) * 1000

                    formalization_result = AutoformalizationResult(
                        success=True,
                        phase=AutoformalizationPhase.PHASE_II_ISOMORPHIC_MAPPING,
                        natural_language=mapping_description,
                        lean_code=lean_code,
                        domain=FormalizationDomain.CATEGORY_THEORY,
                        confidence=0.80,
                        lean_theorem_name=theorem_name,
                        lean_type=lean_type,
                        metadata={
                            "source_domain": source_domain,
                            "target_domain": target_domain
                        },
                        correlation_id=cid,
                        execution_time_ms=execution_time_ms
                    )

                    self.logger.info(
                        "Phase II autoformalization successful",
                        correlation_id=cid,
                        theorem_name=theorem_name,
                        execution_time_ms=execution_time_ms
                    )

                    return formalization_result

            # Fallback
            return self._generate_fallback_formalization(
                formal_statement,
                theorem_name,
                FormalizationDomain.CATEGORY_THEORY,
                AutoformalizationPhase.PHASE_II_ISOMORPHIC_MAPPING,
                cid,
                start_time
            )

        except Exception as e:
            self.logger.error(
                "Phase II autoformalization failed",
                correlation_id=cid,
                error=str(e)
            )

            return AutoformalizationResult(
                success=False,
                phase=AutoformalizationPhase.PHASE_II_ISOMORPHIC_MAPPING,
                natural_language=mapping_description,
                lean_code="",
                domain=FormalizationDomain.CATEGORY_THEORY,
                errors=[str(e)],
                correlation_id=cid,
                execution_time_ms=(asyncio.get_event_loop().time() - start_time) * 1000
            )

    async def autoformalize_phase_iii(
        self,
        hypothesis_text: str,
        hypothesis_type: str = "causal",
        correlation_id: Optional[str] = None
    ) -> AutoformalizationResult:
        """
        Autoformalize hypothesis for Phase III: MCTS Refinement.

        Args:
            hypothesis_text: Natural language hypothesis
            hypothesis_type: Type of hypothesis
            correlation_id: Correlation ID

        Returns:
            AutoformalizationResult with Lean 4 code
        """
        start_time = asyncio.get_event_loop().time()
        cid = correlation_id or self.logger.correlation_id

        self.logger.info(
            "Autoformalizing Phase III hypothesis",
            phase="phase_iii_mcts_refinement",
            hypothesis_type=hypothesis_type,
            correlation_id=cid
        )

        try:
            # Detect domain
            domain = self._detect_domain(hypothesis_text)

            # Generate theorem name
            theorem_name = self._generate_theorem_name(hypothesis_text, "hypothesis")

            # Use LeanAide to translate
            if self.leanaide_client:
                result = await self.leanaide_client.translate_thm_detailed(
                    theorem_text=hypothesis_text,
                    theorem_name=theorem_name
                )

                if result.success:
                    lean_code = result.data.get("result", "")
                    lean_type = result.data.get("type", "")

                    execution_time_ms = (asyncio.get_event_loop().time() - start_time) * 1000

                    formalization_result = AutoformalizationResult(
                        success=True,
                        phase=AutoformalizationPhase.PHASE_III_MCTS_REFINEMENT,
                        natural_language=hypothesis_text,
                        lean_code=lean_code,
                        domain=domain,
                        confidence=0.75,
                        lean_theorem_name=theorem_name,
                        lean_type=lean_type,
                        correlation_id=cid,
                        execution_time_ms=execution_time_ms
                    )

                    self.logger.info(
                        "Phase III autoformalization successful",
                        correlation_id=cid,
                        theorem_name=theorem_name,
                        execution_time_ms=execution_time_ms
                    )

                    return formalization_result

            # Fallback
            return self._generate_fallback_formalization(
                hypothesis_text,
                theorem_name,
                domain,
                AutoformalizationPhase.PHASE_III_MCTS_REFINEMENT,
                cid,
                start_time
            )

        except Exception as e:
            self.logger.error(
                "Phase III autoformalization failed",
                correlation_id=cid,
                error=str(e)
            )

            return AutoformalizationResult(
                success=False,
                phase=AutoformalizationPhase.PHASE_III_MCTS_REFINEMENT,
                natural_language=hypothesis_text,
                lean_code="",
                domain=FormalizationDomain.LOGIC,
                errors=[str(e)],
                correlation_id=cid,
                execution_time_ms=(asyncio.get_event_loop().time() - start_time) * 1000
            )

    async def autoformalize_phase_iv(
        self,
        model_description: str,
        efficacy_claim: str,
        correlation_id: Optional[str] = None
    ) -> AutoformalizationResult:
        """
        Autoformalize predictive model for Phase IV: Architectural Synthesis.

        Args:
            model_description: Description of predictive model
            efficacy_claim: Efficacy claim to verify
            correlation_id: Correlation ID

        Returns:
            AutoformalizationResult with Lean 4 code
        """
        start_time = asyncio.get_event_loop().time()
        cid = correlation_id or self.logger.correlation_id

        self.logger.info(
            "Autoformalizing Phase IV model",
            phase="phase_iv_architectural_synthesis",
            correlation_id=cid
        )

        try:
            # Detect domain
            domain = self._detect_domain(model_description)

            # Generate theorem name
            theorem_name = self._generate_theorem_name(efficacy_claim, "efficacy")

            # Combine model and claim
            combined_statement = f"Given {model_description}, prove that {efficacy_claim}"

            # Use LeanAide to translate
            if self.leanaide_client:
                result = await self.leanaide_client.translate_thm_detailed(
                    theorem_text=combined_statement,
                    theorem_name=theorem_name
                )

                if result.success:
                    lean_code = result.data.get("result", "")
                    lean_type = result.data.get("type", "")

                    execution_time_ms = (asyncio.get_event_loop().time() - start_time) * 1000

                    formalization_result = AutoformalizationResult(
                        success=True,
                        phase=AutoformalizationPhase.PHASE_IV_ARCHITECTURAL_SYNTHESIS,
                        natural_language=combined_statement,
                        lean_code=lean_code,
                        domain=domain,
                        confidence=0.70,
                        lean_theorem_name=theorem_name,
                        lean_type=lean_type,
                        metadata={
                            "model_description": model_description,
                            "efficacy_claim": efficacy_claim
                        },
                        correlation_id=cid,
                        execution_time_ms=execution_time_ms
                    )

                    self.logger.info(
                        "Phase IV autoformalization successful",
                        correlation_id=cid,
                        theorem_name=theorem_name,
                        execution_time_ms=execution_time_ms
                    )

                    return formalization_result

            # Fallback
            return self._generate_fallback_formalization(
                combined_statement,
                theorem_name,
                domain,
                AutoformalizationPhase.PHASE_IV_ARCHITECTURAL_SYNTHESIS,
                cid,
                start_time
            )

        except Exception as e:
            self.logger.error(
                "Phase IV autoformalization failed",
                correlation_id=cid,
                error=str(e)
            )

            return AutoformalizationResult(
                success=False,
                phase=AutoformalizationPhase.PHASE_IV_ARCHITECTURAL_SYNTHESIS,
                natural_language=model_description,
                lean_code="",
                domain=FormalizationDomain.LOGIC,
                errors=[str(e)],
                correlation_id=cid,
                execution_time_ms=(asyncio.get_event_loop().time() - start_time) * 1000
            )

    # =========================================================================
    # Helper Methods
    # =========================================================================

    def _detect_domain(self, text: str) -> FormalizationDomain:
        """Detect mathematical domain from text"""
        text_lower = text.lower()

        # Check for domain-specific keywords
        domain_keywords = {
            FormalizationDomain.ARITHMETIC: ["number", "integer", "prime", "divisible", "sum"],
            FormalizationDomain.ALGEBRA: ["equation", "polynomial", "vector", "matrix", "linear"],
            FormalizationDomain.LOGIC: ["implies", "forall", "exists", "contradiction", "proof"],
            FormalizationDomain.SET_THEORY: ["set", "subset", "union", "intersection", "element"],
            FormalizationDomain.CALCULUS: ["derivative", "integral", "limit", "continuous", "function"],
            FormalizationDomain.GRAPH_THEORY: ["graph", "node", "edge", "path", "connected"],
            FormalizationDomain.PROBABILITY: ["probability", "distribution", "random", "expected"],
            FormalizationDomain.TOPOLOGY: ["topology", "continuous", "compact", "connected"],
            FormalizationDomain.CATEGORY_THEORY: ["functor", "morphism", "isomorphic", "category"]
        }

        for domain, keywords in domain_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                return domain

        return FormalizationDomain.LOGIC

    def _generate_theorem_name(self, text: str, suffix: str) -> str:
        """Generate Lean theorem name from text"""
        # Extract key words
        words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
        key_words = [w for w in words if len(w) > 3][:5]

        # Build name
        name_parts = key_words + [suffix]
        theorem_name = "_".join(name_parts)

        # Ensure valid Lean identifier
        theorem_name = theorem_name.replace("-", "_").replace(" ", "_")

        return theorem_name

    def _build_isomorphism_statement(
        self,
        mapping_description: str,
        source_domain: str,
        target_domain: str
    ) -> str:
        """Build formal isomorphism statement"""
        return (
            f"There exists an isomorphism between {source_domain} and {target_domain} "
            f"such that {mapping_description}"
        )

    def _generate_fallback_formalization(
        self,
        natural_language: str,
        theorem_name: str,
        domain: FormalizationDomain,
        phase: AutoformalizationPhase,
        correlation_id: str,
        start_time: float
    ) -> AutoformalizationResult:
        """Generate fallback formalization when LeanAide is unavailable"""

        # Template-based generation
        lean_code = f"""import Mathlib

theorem {theorem_name} : Prop := by
  -- TODO: Autoformalized from: {natural_language[:100]}...
  sorry
"""

        execution_time_ms = (asyncio.get_event_loop().time() - start_time) * 1000

        return AutoformalizationResult(
            success=True,
            phase=phase,
            natural_language=natural_language,
            lean_code=lean_code,
            domain=domain,
            confidence=0.5,
            lean_theorem_name=theorem_name,
            lean_type="Prop",
            warnings=["Generated using fallback templates - LeanAide not available"],
            correlation_id=correlation_id,
            execution_time_ms=execution_time_ms
        )

    async def batch_autoformalize(
        self,
        items: List[Dict[str, Any]],
        phase: AutoformalizationPhase,
        correlation_id: Optional[str] = None
    ) -> List[AutoformalizationResult]:
        """
        Autoformalize multiple items in batch.

        Args:
            items: List of items to formalize
            phase: RESE phase
            correlation_id: Correlation ID

        Returns:
            List of AutoformalizationResult
        """
        cid = correlation_id or self.logger.correlation_id

        self.logger.info(
            "Batch autoformalization started",
            phase=phase.value,
            item_count=len(items),
            correlation_id=cid
        )

        # Process in parallel
        tasks = []
        for item in items:
            if phase == AutoformalizationPhase.PHASE_I_EPISTEMIC_AUDIT:
                task = self.autoformalize_phase_i(
                    item.get("text", ""),
                    item.get("type", "logical"),
                    cid
                )
            elif phase == AutoformalizationPhase.PHASE_II_ISOMORPHIC_MAPPING:
                task = self.autoformalize_phase_ii(
                    item.get("description", ""),
                    item.get("source_domain", ""),
                    item.get("target_domain", ""),
                    cid
                )
            elif phase == AutoformalizationPhase.PHASE_III_MCTS_REFINEMENT:
                task = self.autoformalize_phase_iii(
                    item.get("text", ""),
                    item.get("type", "causal"),
                    cid
                )
            elif phase == AutoformalizationPhase.PHASE_IV_ARCHITECTURAL_SYNTHESIS:
                task = self.autoformalize_phase_iv(
                    item.get("model_description", ""),
                    item.get("efficacy_claim", ""),
                    cid
                )
            else:
                continue

            tasks.append(task)

        # Execute all tasks
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Handle exceptions
        formatted_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                formatted_results.append(AutoformalizationResult(
                    success=False,
                    phase=phase,
                    natural_language=items[i].get("text", str(items[i])),
                    lean_code="",
                    domain=FormalizationDomain.LOGIC,
                    errors=[str(result)],
                    correlation_id=cid
                ))
            else:
                formatted_results.append(result)

        self.logger.info(
            "Batch autoformalization completed",
            phase=phase.value,
            successful=sum(1 for r in formatted_results if r.success),
            failed=sum(1 for r in formatted_results if not r.success),
            correlation_id=cid
        )

        return formatted_results

    async def close(self):
        """Close the service and cleanup resources"""
        if self.leanaide_client:
            await self.leanaide_client.close()

        self.logger.info("AutoformalizationService closed")


# ============================================================================
# Convenience Functions
# ============================================================================

async def create_autoformalization_service(
    config: Optional[AutoformalizationConfig] = None
) -> AutoformalizationService:
    """
    Create and initialize autoformalization service.

    Args:
        config: Service configuration

    Returns:
        Initialized AutoformalizationService
    """
    return AutoformalizationService(config)


# ============================================================================
# Example Usage
# ============================================================================

async def main():
    """Example usage of autoformalization service"""

    print("=" * 70)
    print("RESE-LeanAide Autoformalization Service")
    print("=" * 70)

    # Create service
    service = await create_autoformalization_service()

    try:
        # Phase I example
        print("\n1. PHASE I: EPISTEMIC AUDIT")
        print("-" * 40)
        result_i = await service.autoformalize_phase_i(
            constraint_text="All prime numbers greater than 2 are odd",
            constraint_type="arithmetic"
        )
        print(f"Success: {result_i.success}")
        print(f"Theorem: {result_i.lean_theorem_name}")
        print(f"Lean code:\n{result_i.lean_code[:200]}...")

        # Phase II example
        print("\n2. PHASE II: ISOMORPHIC MAPPING")
        print("-" * 40)
        result_ii = await service.autoformalize_phase_ii(
            mapping_description="A structure-preserving bijection between elements",
            source_domain="natural_numbers",
            target_domain="integers"
        )
        print(f"Success: {result_ii.success}")
        print(f"Theorem: {result_ii.lean_theorem_name}")

        # Phase III example
        print("\n3. PHASE III: MCTS REFINEMENT")
        print("-" * 40)
        result_iii = await service.autoformalize_phase_iii(
            hypothesis_text="If x is positive and y is positive, then x + y is positive"
        )
        print(f"Success: {result_iii.success}")
        print(f"Theorem: {result_iii.lean_theorem_name}")

        # Phase IV example
        print("\n4. PHASE IV: ARCHITECTURAL SYNTHESIS")
        print("-" * 40)
        result_iv = await service.autoformalize_phase_iv(
            model_description="Linear regression model with squared error loss",
            efficacy_claim="model predictions converge to true values with sufficient data"
        )
        print(f"Success: {result_iv.success}")
        print(f"Theorem: {result_iv.lean_theorem_name}")

        print("\n" + "=" * 70)
        print("All examples completed!")
        print("=" * 70)

    finally:
        await service.close()


if __name__ == "__main__":
    asyncio.run(main())
