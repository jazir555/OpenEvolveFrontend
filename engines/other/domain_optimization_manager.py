"""
Domain Optimization Manager

Manages domain-specific optimizations for decomposition, including
domain configurations, patterns, vocabulary, and strategy adjustments.
"""
from __future__ import annotations


import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
from dataclasses import dataclass, field

from domain_configurations import (
    DomainConfiguration,
    get_domain_config,
    get_all_domains,
    register_domain_config
)
from sovereign_data_models import (
    DecompositionPlan,
    SubProblem,
    ProblemDefinition,
    EnhancedDomainContext
)

# **ACTUAL INTEGRATION**: Alerting, knowledge, and adaptive for Domain Optimization Manager
try:
    from alerting_system import get_alert_manager, AlertSeverity
    ALERTING_AVAILABLE = True
except ImportError:
    ALERTING_AVAILABLE = False

try:
    from knowledge_engine.enterprise_knowledge_engine import get_knowledge_engine, KnowledgeArtifact
    KNOWLEDGE_AVAILABLE = True
except ImportError:
    KNOWLEDGE_AVAILABLE = False

try:
    from adaptive_strategy_selector import StrategyPerformanceTracker, StrategyPerformanceData
    ADAPTIVE_AVAILABLE = True
except ImportError:
    ADAPTIVE_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class DomainPattern:
    """Pattern observed in a domain."""
    pattern_id: str
    domain: str
    pattern_name: str
    description: str

    # Evidence
    support_count: int = 1  # How many times observed
    success_rate: float = 0.5  # 0-1

    # Application
    when_applies: str = ""
    how_to_use: str = ""

    # Metadata
    discovered_at: datetime = field(default_factory=datetime.now)
    last_observed: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = {
            'pattern_id': self.pattern_id,
            'domain': self.domain,
            'pattern_name': self.pattern_name,
            'description': self.description,
            'support_count': self.support_count,
            'success_rate': self.success_rate,
            'when_applies': self.when_applies,
            'how_to_use': self.how_to_use,
            'discovered_at': self.discovered_at.isoformat(),
            'last_observed': self.last_observed.isoformat()
        }
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DomainPattern':
        """Create from dictionary."""
        data = data.copy()
        data['discovered_at'] = datetime.fromisoformat(data['discovered_at'])
        data['last_observed'] = datetime.fromisoformat(data['last_observed'])
        return cls(**data)


@dataclass
class DomainTerm:
    """Single term in domain vocabulary."""
    term: str
    definition: str
    examples: List[str] = field(default_factory=list)
    related_terms: List[str] = field(default_factory=list)
    category: str = "concept"  # "concept", "tool", "process", "metric"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'term': self.term,
            'definition': self.definition,
            'examples': self.examples,
            'related_terms': self.related_terms,
            'category': self.category
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DomainTerm':
        """Create from dictionary."""
        return cls(**data)


@dataclass
class DomainVocabulary:
    """Domain-specific vocabulary."""
    domain: str
    terms: Dict[str, DomainTerm] = field(default_factory=dict)
    abbreviations: Dict[str, str] = field(default_factory=dict)  # abbreviation -> full term
    acronyms: Dict[str, str] = field(default_factory=dict)  # acronym -> meaning

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'domain': self.domain,
            'terms': {k: v.to_dict() for k, v in self.terms.items()},
            'abbreviations': self.abbreviations,
            'acronyms': self.acronyms
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DomainVocabulary':
        """Create from dictionary."""
        data = data.copy()
        data['terms'] = {k: DomainTerm.from_dict(v) for k, v in data['terms'].items()}
        return cls(**data)


class DomainOptimizationManager:
    """
    Manages domain-specific optimizations for decomposition.

    Each domain has unique patterns, terminology, and approaches.
    Optimizations include:
    - Domain-specific terminology
    - Common patterns in domain
    - Typical complexity distribution
    - Preferred strategies
    - Team expertise mapping
    """

    def __init__(self):
        """Initialize with domain configurations."""
        self.domain_configs: Dict[str, DomainConfiguration] = {}

        # Load predefined configurations
        for domain in get_all_domains():
            self.domain_configs[domain] = get_domain_config(domain)

        # Learnable patterns
        self.domain_patterns: Dict[str, List[DomainPattern]] = {}

        # Extended vocabulary
        self.domain_vocabularies: Dict[str, DomainVocabulary] = {}

        logger.info(f"DomainOptimizationManager initialized with {len(self.domain_configs)} domains")

    def get_domain_config(self, domain: str) -> Optional[DomainConfiguration]:
        """
        Get configuration for domain.

        Args:
            domain: Domain identifier

        Returns:
            DomainConfiguration or None if not found
        """
        return self.domain_configs.get(domain)

    def get_recommended_strategies(
        self,
        domain: str,
        problem_type: str = None
    ) -> List[str]:
        """
        Get recommended strategies for domain.

        Args:
            domain: Domain identifier
            problem_type: Optional problem type for filtering

        Returns:
            List of recommended strategy names (ordered by priority)
        """
        config = self.get_domain_config(domain)
        if not config:
            # Return default strategies
            return ["functional", "technical_dependency", "complexity"]

        # Use strategy weights to order strategies
        strategies = list(config.strategy_weights.keys())
        strategies.sort(key=lambda s: config.strategy_weights.get(s, 0), reverse=True)

        # Filter out avoided strategies
        strategies = [s for s in strategies if s not in config.avoided_strategies]

        return strategies

    def get_domain_vocabulary(self, domain: str) -> DomainVocabulary:
        """
        Get domain-specific vocabulary.

        Args:
            domain: Domain identifier

        Returns:
            DomainVocabulary for the domain
        """
        if domain not in self.domain_vocabularies:
            # Create vocabulary from domain config
            config = self.get_domain_config(domain)
            if config:
                vocab = DomainVocabulary(domain=domain)

                # Add terminology
                for term, definition in config.terminology.items():
                    vocab.terms[term] = DomainTerm(
                        term=term,
                        definition=definition,
                        category="concept"
                    )

                self.domain_vocabularies[domain] = vocab
            else:
                # Return empty vocabulary
                self.domain_vocabularies[domain] = DomainVocabulary(domain=domain)

        return self.domain_vocabularies[domain]

    def optimize_decomposition(
        self,
        plan: DecompositionPlan,
        domain: str
    ) -> DecompositionPlan:
        """
        Optimize decomposition for domain.

        Applies:
        - Domain-specific terminology
        - Common domain patterns
        - Optimal strategy adjustments
        - Domain-specific quality thresholds

        Args:
            plan: Original decomposition plan
            domain: Domain identifier

        Returns:
            Optimized decomposition plan
        """
        import time
        start_time = time.time()

        try:
            config = self.get_domain_config(domain)
            if not config:
                logger.warning(f"No configuration found for domain '{domain}', returning original plan")
                return plan

            logger.info(f"Optimizing decomposition plan for domain: {domain}")

            # Get vocabulary
            vocabulary = self.get_domain_vocabulary(domain)

            # Optimize each sub-problem
            optimized_sub_problems = []
            for sub_problem in plan.sub_problems:
                optimized_sub = self._optimize_sub_problem(sub_problem, config, vocabulary)
                optimized_sub_problems.append(optimized_sub)

            # Create optimized plan
            optimized_plan = DecompositionPlan(
                id=plan.id,
                problem_id=plan.problem_id,
                strategy=plan.strategy,
                sub_problems=optimized_sub_problems,
                dependency_graph=plan.dependency_graph,
                validation_checkpoints=plan.validation_checkpoints,
                quality_scores=plan.quality_scores,
                enhanced_quality_scores=plan.enhanced_quality_scores,
                confidence_level=plan.confidence_level,
                created_by=plan.created_by,
                approved_by=plan.approved_by,
                status=plan.status,
                created_at=plan.created_at,
                updated_at=datetime.now(),
                metadata={
                    **plan.metadata,
                    'domain_optimized': True,
                    'domain': domain,
                    'optimization_applied': True
                }
            )

            logger.info(f"Domain optimization complete for {len(optimized_sub_problems)} sub-problems")

            # **ACTUAL INTEGRATION**: Extract knowledge and track performance for successful optimization
            duration = time.time() - start_time
            self._extract_domain_optimization_knowledge("optimize_decomposition", domain, plan, optimized_plan)
            self._track_domain_optimization_performance("optimize_decomposition", True, duration, domain, len(optimized_sub_problems))

            return optimized_plan

        except (RuntimeError, ValueError, TypeError) as e:
            logger.error(f"Error optimizing decomposition for domain '{domain}': {e}", exc_info=True)
            duration = time.time() - start_time

            # **ACTUAL INTEGRATION**: Trigger alert and track failure
            self._trigger_domain_optimization_alerts("optimize_decomposition", False, domain, plan.id, str(e))
            self._track_domain_optimization_performance("optimize_decomposition", False, duration, domain, 0)

            raise

    def _optimize_sub_problem(
        self,
        sub_problem: SubProblem,
        config: DomainConfiguration,
        vocabulary: DomainVocabulary
    ) -> SubProblem:
        """
        Optimize a single sub-problem for domain.

        Args:
            sub_problem: Original sub-problem
            config: Domain configuration
            vocabulary: Domain vocabulary

        Returns:
            Optimized sub-problem
        """
        # Enhance description with domain terminology
        optimized_description = self._enhance_with_terminology(
            sub_problem.description,
            vocabulary
        )

        # Adjust complexity based on domain typical complexity
        adjusted_complexity = self._adjust_complexity(
            sub_problem.complexity_score,
            config
        )

        # Add domain-specific expertise
        enhanced_expertise = list(set(
            sub_problem.required_expertise + config.required_expertise
        ))

        # Adjust estimated effort based on domain multipliers
        adjusted_effort = self._adjust_effort(
            sub_problem.estimated_effort,
            sub_problem.type.value if hasattr(sub_problem.type, 'value') else str(sub_problem.type),
            config
        )

        # Create optimized sub-problem
        optimized_sub = SubProblem(
            id=sub_problem.id,
            parent_id=sub_problem.parent_id,
            title=sub_problem.title,
            description=optimized_description,
            type=sub_problem.type,
            complexity_score=adjusted_complexity,
            dependencies=sub_problem.dependencies,
            success_criteria=sub_problem.success_criteria,
            validation_gauntlet=sub_problem.validation_gauntlet,
            assigned_team=sub_problem.assigned_team,
            estimated_effort=adjusted_effort,
            priority=sub_problem.priority,
            status=sub_problem.status,
            created_at=sub_problem.created_at,
            updated_at=datetime.now(),
            solution_attempts=sub_problem.solution_attempts,
            metadata={
                **sub_problem.metadata,
                'domain_optimized': True,
                'original_description': sub_problem.description
            },
            # Enhanced fields
            acceptance_criteria=sub_problem.acceptance_criteria,
            ai_suggested_evolution_mode=sub_problem.ai_suggested_evolution_mode,
            ai_suggested_complexity_score=sub_problem.ai_suggested_complexity_score,
            ai_suggested_evaluation_prompt=sub_problem.ai_suggested_evaluation_prompt,
            ai_suggested_team_assignment=sub_problem.ai_suggested_team_assignment,
            ai_suggested_gauntlet_assignment=sub_problem.ai_suggested_gauntlet_assignment,
            estimated_resources=sub_problem.estimated_resources,
            potential_approaches=sub_problem.potential_approaches,
            required_expertise=enhanced_expertise,
            associated_risks=sub_problem.associated_risks,
            success_dependencies=sub_problem.success_dependencies,
            testing_approach=sub_problem.testing_approach,
            quality_metrics=sub_problem.quality_metrics
        )

        return optimized_sub

    def _enhance_with_terminology(
        self,
        text: str,
        vocabulary: DomainVocabulary
    ) -> str:
        """
        Enhance text with domain terminology.

        Args:
            text: Original text
            vocabulary: Domain vocabulary

        Returns:
            Enhanced text with domain-specific terminology
        """
        # This is a simple implementation
        # In production, you'd use LLM to intelligently incorporate terminology

        # Add domain context to metadata
        enhanced = text

        # Note: In a full implementation, you might:
        # - Use LLM to rewrite with domain terminology
        # - Add domain-specific context
        # - Include relevant patterns

        return enhanced

    def _adjust_complexity(
        self,
        original_complexity: Any,
        config: DomainConfiguration
    ) -> Any:
        """
        Adjust complexity score based on domain typical complexity.

        Args:
            original_complexity: Original complexity score
            config: Domain configuration

        Returns:
            Adjusted complexity score
        """
        # Import here to avoid circular import
        from sovereign_data_models import ComplexityScore

        if not isinstance(original_complexity, ComplexityScore):
            return original_complexity

        # Adjust overall complexity towards domain typical
        typical = config.typical_complexity * 10  # Convert to 0-10 scale
        adjustment_factor = 0.1  # Adjust 10% towards typical

        adjusted_overall = original_complexity.overall_complexity * (1 - adjustment_factor) + typical * adjustment_factor

        # Create adjusted complexity
        return ComplexityScore(
            explanation=original_complexity.explanation,
            cognitive_complexity=original_complexity.cognitive_complexity,
            computational_complexity=original_complexity.computational_complexity,
            domain_complexity=original_complexity.domain_complexity,
            integration_complexity=original_complexity.integration_complexity,
            overall_complexity=adjusted_overall,
            metadata={
                **original_complexity.metadata,
                'domain_adjusted': True,
                'original_overall': original_complexity.overall_complexity
            }
        )

    def _adjust_effort(
        self,
        original_effort: int,
        problem_type: str,
        config: DomainConfiguration
    ) -> int:
        """
        Adjust estimated effort based on domain multipliers.

        Args:
            original_effort: Original effort estimate (person-hours)
            problem_type: Type of problem
            config: Domain configuration

        Returns:
            Adjusted effort estimate
        """
        # Get multiplier for problem type
        multiplier = config.typical_effort_multipliers.get(problem_type, 1.0)

        # Also apply generic resource multiplier
        resource_multiplier = config.resource_multipliers.get("time_hours", 1.0)

        # Calculate adjusted effort
        adjusted = int(original_effort * multiplier * resource_multiplier)

        return max(1, adjusted)  # At least 1 hour

    def add_domain_pattern(
        self,
        domain: str,
        pattern: DomainPattern
    ):
        """
        Add discovered pattern to domain configuration.

        Args:
            domain: Domain identifier
            pattern: Pattern to add
        """
        if domain not in self.domain_patterns:
            self.domain_patterns[domain] = []

        self.domain_patterns[domain].append(pattern)

        logger.info(f"Added pattern '{pattern.pattern_name}' to domain '{domain}'")

    def get_domain_patterns(self, domain: str) -> List[DomainPattern]:
        """
        Get patterns for a domain.

        Args:
            domain: Domain identifier

        Returns:
            List of domain patterns
        """
        return self.domain_patterns.get(domain, [])

    def enhance_domain_context(
        self,
        problem: ProblemDefinition,
        domain: str
    ) -> EnhancedDomainContext:
        """
        Create enhanced domain context for a problem.

        Args:
            problem: Problem definition
            domain: Domain identifier

        Returns:
            EnhancedDomainContext with domain-specific information
        """
        config = self.get_domain_config(domain)
        vocabulary = self.get_domain_vocabulary(domain)
        patterns = self.get_domain_patterns(domain)

        # Extract key concepts from terminology
        key_concepts = list(vocabulary.terms.keys())

        # Build concept relationships from related terms
        concept_relationships = {}
        for term, domain_term in vocabulary.terms.items():
            concept_relationships[term] = domain_term.related_terms

        # Create semantic clusters (simple grouping by category)
        semantic_clusters = []
        categories = {}
        for term, domain_term in vocabulary.terms.items():
            if domain_term.category not in categories:
                categories[domain_term.category] = []
            categories[domain_term.category].append(term)
        semantic_clusters = list(categories.values())

        # Create enhanced context
        enhanced_context = EnhancedDomainContext(
            domain=domain,
            subdomain=problem.domain_context.subdomain,
            related_domains=problem.domain_context.related_domains,
            domain_knowledge=problem.domain_context.domain_knowledge,
            key_concepts=key_concepts,
            concept_relationships=concept_relationships,
            semantic_clusters=semantic_clusters,
            terminology=vocabulary.abbreviations,
            domain_complexity=config.typical_complexity if config else 0.5,
            abstraction_level="medium",
            typical_decomposition_approach=config.preferred_strategies[0] if config else "hybrid",
            similar_problems=[],
            domain_patterns=config.common_patterns if config else [],
            best_practices=config.decomposition_approaches if config else [],
            context_sources=["domain_config", "vocabulary"],
            confidence_score=0.8,
            metadata={
                **problem.domain_context.metadata,
                'domain_enhanced': True,
                'vocabulary_terms': len(vocabulary.terms),
                'patterns_count': len(patterns)
            }
        )

        return enhanced_context

    # =========================================================================
    # ACTUAL INTEGRATION METHODS - Alerting, knowledge, and adaptive for Domain Optimization Manager
    # =========================================================================

    def _trigger_domain_optimization_alerts(
        self,
        operation: str,
        success: bool,
        domain: str,
        plan_id: Optional[str] = None,
        error: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """**ACTUAL INTEGRATION**: Trigger alerts for domain optimization failures."""
        if not ALERTING_AVAILABLE:
            return

        try:
            alert_manager = get_alert_manager()

            # Alert on failures
            if not success:
                alert_manager.create_alert(
                    title=f"Domain Optimization Alert: {operation}",
                    description=f"Domain optimization operation '{operation}' failed for domain '{domain}'" +
                                 (f" on plan '{plan_id}'" if plan_id else "") +
                                 ". " + (f"Error: {error}" if error else ""),
                    severity=AlertSeverity.MEDIUM.value,
                    source="domain_optimization_manager",
                    component="domain_optimization",
                    metadata=metadata or {}
                )

        except Exception as e:
            logger.error(f"Failed to trigger Domain Optimization alert: {e}")

    def _extract_domain_optimization_knowledge(
        self,
        operation: str,
        domain: str,
        plan: DecompositionPlan,
        optimized_plan: DecompositionPlan
    ) -> bool:
        """**ACTUAL INTEGRATION**: Extract domain optimization knowledge to knowledge engine."""
        if not KNOWLEDGE_AVAILABLE:
            return False

        try:
            knowledge_engine = get_knowledge_engine()

            artifact = KnowledgeArtifact(
                artifact_id=f"domain_opt_{operation}_{domain}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                artifact_type="domain_optimization",
                source_component="domain_optimization_manager",
                title=f"Domain Optimization: {domain} ({operation})",
                content={
                    "operation": operation,
                    "domain": domain,
                    "plan_id": plan.id,
                    "optimized_plan_id": optimized_plan.id,
                    "original_sub_problems": len(plan.sub_problems),
                    "optimized_sub_problems": len(optimized_plan.sub_problems),
                    "strategy": plan.strategy,
                    "timestamp": datetime.now().isoformat()
                },
                metadata={
                    "domain": domain,
                    "original_confidence": plan.confidence_level,
                    "optimized_confidence": optimized_plan.confidence_level
                },
                tags=["domain_optimization", domain, operation, "decomposition"]
            )

            knowledge_engine.store_artifact(artifact)
            logger.debug(f"Extracted domain optimization knowledge for {domain}")
            return True

        except Exception as e:
            logger.error(f"Failed to extract domain optimization knowledge: {e}")
            return False

    def _track_domain_optimization_performance(
        self,
        operation: str,
        success: bool,
        duration_seconds: float,
        domain: str,
        sub_problems_count: int = 0
    ):
        """**ACTUAL INTEGRATION**: Track domain optimization performance in adaptive selector."""
        if not ADAPTIVE_AVAILABLE:
            return

        try:
            tracker = StrategyPerformanceTracker()

            quality = 1.0 if success else 0.0

            performance_data = StrategyPerformanceData(
                strategy_name=f"domain_optimization_{operation}_{domain}",
                success_count=1 if success else 0,
                failure_count=0 if success else 1,
                average_quality=quality,
                last_used=datetime.now(),
                total_attempts=1,
                metadata={
                    "operation": operation,
                    "duration_seconds": duration_seconds,
                    "domain": domain,
                    "sub_problems_count": sub_problems_count
                }
            )

            if hasattr(tracker, 'performance_history'):
                tracker.performance_history.append(performance_data)
                logger.debug(f"Tracked domain optimization performance for {domain}")

        except Exception as e:
            logger.error(f"Failed to track domain optimization performance: {e}")
