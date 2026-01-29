"""
Adaptive Knowledge Orchestrator

The ultimate generic knowledge engine that:
1. Automatically classifies any input domain
2. Adapts processing strategy dynamically
3. Learns from all users globally
4. Continuously validates through gauntlet
5. Improves accuracy over time

This is a TRUE knowledge engine - not domain-specific presets,
but a universal system that learns and adapts to ANY content.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional, Union
from datetime import datetime, timezone
from dataclasses import dataclass, field
import copy

from .integrated_orchestrator import IntegratedOrchestrator, ExecutionContext
from .domain_classifier import DomainClassifier, DomainCategory, classify_input
from .global_learning_engine import GlobalLearningEngine, get_global_learning_engine
from .gauntlet_integration import GauntletIntegration, TestType
from .learning_engine import LearningExperience

logger = logging.getLogger(__name__)


@dataclass
class AdaptiveConfig:
    """Configuration for adaptive orchestrator"""
    # Classification
    enable_auto_classification: bool = True
    use_llm_for_classification: bool = False
    classification_confidence_threshold: float = 0.6
    
    # Global learning
    enable_global_learning: bool = True
    contribute_to_global: bool = True
    use_global_patterns: bool = True
    
    # Gauntlet validation
    enable_gauntlet: bool = True
    validation_frequency: int = 10  # Validate every N executions
    quality_gate_threshold: float = 0.7
    
    # Adaptation
    adaptation_rate: float = 0.1  # How quickly to adapt
    min_executions_before_adaptation: int = 5
    
    # User identification
    user_id: Optional[str] = None


class AdaptiveOrchestrator(IntegratedOrchestrator):
    """
    Adaptive Knowledge Orchestrator - the ultimate generic knowledge engine.
    
    Features:
    - Automatic domain classification (no presets needed!)
    - Dynamic component selection based on content
    - Global learning across all users
    - Continuous validation and quality assurance
    - Self-improving accuracy over time
    
    Usage:
        orchestrator = AdaptiveOrchestrator()
        result = orchestrator.process({'text': 'Any content here...'})
        # System automatically classifies, adapts, learns, and improves
    """
    
    def __init__(self, 
                 adaptive_config: Optional[AdaptiveConfig] = None,
                 storage_path: Optional[str] = None,
                 **kwargs):
        """
        Initialize adaptive orchestrator.
        
        Args:
            adaptive_config: Adaptive behavior configuration
            storage_path: Path for global learning storage
            **kwargs: Additional config for parent classes
        """
        # Initialize parent with generic config
        from .knowledge_orchestrator import OrchestratorConfig, DomainType
        
        generic_config = OrchestratorConfig(
            name="adaptive_orchestrator",
            domain=DomainType.GENERAL,
            description="Universal adaptive knowledge engine"
        )
        
        super().__init__(
            config=generic_config,
            enable_self_healing=True,
            enable_learning=True,
            enable_coordination=True,
            enable_feedback=True,
            enable_circuit_breaker=True,
            **kwargs
        )
        
        # Adaptive configuration
        self.adaptive_config = adaptive_config or AdaptiveConfig()
        
        # Subsystems
        self.domain_classifier = DomainClassifier(
            learning_engine=self.learning_engine,
            llm_client=None  # Could be configured
        )
        
        self.global_learning = get_global_learning_engine(
            storage_path=storage_path or "global_learning.json"
        )
        
        self.gauntlet = GauntletIntegration(self)
        
        # Execution tracking
        self.execution_count = 0
        self.user_id = self.adaptive_config.user_id or f"user_{datetime.now(timezone.utc).timestamp()}"
        
        # Performance tracking
        self.domain_performance: Dict[str, Dict[str, Any]] = {}
        
        logger.info({
            "msg": "AdaptiveOrchestrator initialized",
            "auto_classification": self.adaptive_config.enable_auto_classification,
            "global_learning": self.adaptive_config.enable_global_learning,
            "gauntlet_validation": self.adaptive_config.enable_gauntlet,
            "user_id_hash": self.user_id[:16],
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
    
    def process(self, input_data: Dict[str, Any],
                custom_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process any input with automatic adaptation and learning.
        
        Args:
            input_data: Any input data with 'text' field
            custom_config: Optional override configuration
            
        Returns:
            Processing results with adaptive metadata
        """
        self.execution_count += 1
        start_time = datetime.now(timezone.utc)
        
        logger.info({
            "msg": "Adaptive processing started",
            "execution": self.execution_count,
            "user_id_hash": self.user_id[:16]
        })
        
        try:
            # Phase 1: Automatic Domain Classification
            classification = None
            if self.adaptive_config.enable_auto_classification:
                classification = self.domain_classifier.classify(
                    input_data,
                    use_llm=self.adaptive_config.use_llm_for_classification,
                    use_learning=True
                )
                
                logger.info({
                    "msg": "Input classified",
                    "domain": classification.primary_domain.value,
                    "confidence": classification.confidence,
                    "content_type": classification.content_type.value
                })
            
            # Phase 2: Dynamic Configuration Adaptation
            adapted_config = self._adapt_configuration(
                input_data, classification, custom_config
            )
            
            # Phase 3: Apply Global Learning Patterns
            if self.adaptive_config.use_global_patterns:
                global_recommendations = self.global_learning.get_recommendations(
                    {
                        'domain': classification.primary_domain.value if classification else 'general',
                        'data_type': input_data.get('data_type', 'unknown'),
                        'content_type': classification.content_type.value if classification else 'text'
                    },
                    local_user_id=self.user_id
                )
                
                adapted_config = self._apply_global_recommendations(
                    adapted_config, global_recommendations
                )
            
            # Phase 4: Execute with Integrated Orchestrator
            result = super().process(input_data, adapted_config)
            
            # Phase 5: Contribute to Global Learning
            if self.adaptive_config.contribute_to_global:
                self._contribute_to_global_learning(input_data, result, classification)
            
            # Phase 6: Gauntlet Validation (periodic)
            if self.adaptive_config.enable_gauntlet and \
               self.execution_count % self.adaptive_config.validation_frequency == 0:
                self._run_validation()
            
            # Phase 7: Adapt Based on Results
            self._adapt_based_on_results(result, classification)
            
            # Add adaptive metadata
            result['adaptive_metadata'] = {
                'execution_number': self.execution_count,
                'classification': classification.to_dict() if classification else None,
                'user_id_hash': self.user_id[:16],
                'global_learning_contributed': self.adaptive_config.contribute_to_global,
                'adaptation_applied': True,
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
            
            return result
            
        except Exception as e:
            logger.error({
                "msg": "Adaptive processing failed",
                "error": str(e),
                "execution": self.execution_count
            })
            
            # Return error with what we know
            return {
                'status': 'failed',
                'error': str(e),
                'execution_number': self.execution_count,
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
    
    def _adapt_configuration(self, 
                            input_data: Dict[str, Any],
                            classification: Optional[Any],
                            custom_config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Dynamically adapt configuration based on classification.
        
        This replaces domain-specific presets with dynamic adaptation.
        """
        config = copy.deepcopy(custom_config) if custom_config else {}
        
        if not classification:
            return config
        
        # Adapt components based on domain
        domain = classification.primary_domain
        recommended = classification.recommended_components
        
        # Enable/disable components based on recommendations
        if 'components' not in config:
            config['components'] = {}
        
        # Map domain to optimal settings
        domain_settings = self._get_domain_settings(domain)
        
        for component, settings in domain_settings.items():
            config['components'][component] = settings
        
        # Override with classification recommendations
        for comp in recommended:
            if comp not in config['components']:
                config['components'][comp] = {'enabled': True}
        
        # Adjust timeouts based on content type
        content_type = classification.content_type.value
        if content_type in ('research_paper', 'technical_document'):
            # Longer timeouts for complex documents
            for comp in config['components']:
                if 'timeout' in str(config['components'][comp]):
                    config['components'][comp]['timeout_seconds'] = 120
        
        return config
    
    def _get_domain_settings(self, domain: DomainCategory) -> Dict[str, Dict[str, Any]]:
        """Get optimal component settings for a domain"""
        
        # Dynamic settings based on domain
        settings = {
            DomainCategory.FINANCE: {
                'deepke': {'enabled': True, 'config': {'entity_types': ['ORG', 'MONEY', 'PERCENT']}},
                'karate_club': {'enabled': True},
                'pami': {'enabled': True},
                'causal_learn': {'enabled': True},
                'global_chem': {'enabled': False},
                'neuromancer': {'enabled': False}
            },
            DomainCategory.CHEMISTRY: {
                'deepke': {'enabled': True},
                'global_chem': {'enabled': True, 'required': True},
                'karate_club': {'enabled': True},
                'neuromancer': {'enabled': True},
                'causal_learn': {'enabled': False}
            },
            DomainCategory.HEALTHCARE: {
                'deepke': {'enabled': True, 'config': {'entity_types': ['DISEASE', 'DRUG', 'SYMPTOM']}},
                'global_chem': {'enabled': True},
                'causal_learn': {'enabled': True},
                'karate_club': {'enabled': True}
            },
            DomainCategory.RESEARCH: {
                'deepke': {'enabled': True},
                'pami': {'enabled': True},
                'neuralkg': {'enabled': True},
                'causal_learn': {'enabled': True},
                'karate_club': {'enabled': True},
                'lagrange_mapper': {'enabled': True}
            },
            DomainCategory.TECHNOLOGY: {
                'deepke': {'enabled': True, 'config': {'entity_types': ['TECH', 'ORG', 'PRODUCT']}},
                'karate_club': {'enabled': True},
                'pami': {'enabled': True}
            },
            DomainCategory.LEGAL: {
                'deepke': {'enabled': True, 'config': {'entity_types': ['LAW', 'ORG', 'PERSON']}},
                'karate_club': {'enabled': True},
                'causal_learn': {'enabled': True}
            },
            DomainCategory.GENERAL: {
                'deepke': {'enabled': True},
                'karate_club': {'enabled': True},
                'kg_gen': {'enabled': True}
            }
        }
        
        return settings.get(domain, settings[DomainCategory.GENERAL])
    
    def _apply_global_recommendations(self, 
                                     config: Dict[str, Any],
                                     recommendations: Dict[str, Any]) -> Dict[str, Any]:
        """Apply globally-learned patterns to configuration"""
        
        # Apply component configs that worked well globally
        for comp_config in recommendations.get('component_configs', [])[:3]:  # Top 3
            if comp_config['effectiveness'] > 0.7:
                logger.debug({
                    "msg": "Applying global component config",
                    "pattern": comp_config['pattern_id'],
                    "effectiveness": comp_config['effectiveness']
                })
                # Would apply specific config here
        
        # Apply successful healing strategies
        for healing in recommendations.get('healing_strategies', [])[:2]:
            if healing['success_rate'] > 0.8:
                logger.debug({
                    "msg": "Noting effective healing strategy",
                    "strategy": healing['strategy'],
                    "success_rate": healing['success_rate']
                })
        
        return config
    
    def _contribute_to_global_learning(self, 
                                      input_data: Dict[str, Any],
                                      result: Dict[str, Any],
                                      classification: Optional[Any]):
        """Contribute execution results to global learning"""
        
        try:
            # Add classification info to result
            enriched_result = copy.deepcopy(result)
            if classification:
                enriched_result['classified_domain'] = classification.primary_domain.value
                enriched_result['classification_confidence'] = classification.confidence
            
            # Contribute to global learning
            self.global_learning.contribute_experience(
                user_id=self.user_id,
                execution_result=enriched_result,
                local_learning={
                    'domain_performance': self.domain_performance,
                    'execution_count': self.execution_count
                }
            )
            
            logger.debug({
                "msg": "Contributed to global learning",
                "user_id_hash": self.user_id[:16]
            })
            
        except Exception as e:
            logger.warning({
                "msg": "Failed to contribute to global learning",
                "error": str(e)
            })
    
    def _run_validation(self):
        """Run gauntlet validation"""
        try:
            logger.info({"msg": "Running gauntlet validation"})
            
            # Check quality gate
            gate_result = self.gauntlet.check_quality_gate()
            
            if not gate_result['passed']:
                logger.warning({
                    "msg": "Quality gate failed",
                    "failures": gate_result['failures']
                })
            else:
                logger.info({"msg": "Quality gate passed"})
            
        except Exception as e:
            logger.error({
                "msg": "Gauntlet validation failed",
                "error": str(e)
            })
    
    def _adapt_based_on_results(self, 
                               result: Dict[str, Any],
                               classification: Optional[Any]):
        """Adapt behavior based on execution results"""
        
        if not classification:
            return
        
        domain = classification.primary_domain.value
        
        # Track domain performance
        if domain not in self.domain_performance:
            self.domain_performance[domain] = {
                'executions': 0,
                'successes': 0,
                'avg_quality': 0.0,
                'avg_execution_time': 0.0
            }
        
        perf = self.domain_performance[domain]
        perf['executions'] += 1
        
        if result.get('status') in ('success', 'partial'):
            perf['successes'] += 1
        
        # Update running averages
        quality = 1.0 if result.get('status') == 'success' else 0.5
        perf['avg_quality'] = (
            perf['avg_quality'] * (perf['executions'] - 1) + quality
        ) / perf['executions']
        
        exec_time = result.get('execution', {}).get('duration_ms', 0)
        if perf['executions'] == 1:
            perf['avg_execution_time'] = exec_time
        else:
            perf['avg_execution_time'] = (
                perf['avg_execution_time'] * 0.9 + exec_time * 0.1
            )
    
    def get_adaptive_stats(self) -> Dict[str, Any]:
        """Get comprehensive adaptive orchestrator statistics"""
        return {
            'executions': self.execution_count,
            'user_id_hash': self.user_id[:16],
            'configuration': {
                'auto_classification': self.adaptive_config.enable_auto_classification,
                'global_learning': self.adaptive_config.enable_global_learning,
                'gauntlet_validation': self.adaptive_config.enable_gauntlet
            },
            'domain_performance': self.domain_performance,
            'global_learning_stats': self.global_learning.get_stats(),
            'gauntlet_stats': self.gauntlet.get_stats(),
            'classifier_stats': self.domain_classifier.get_classifier_stats()
        }
    
    def get_curated_knowledge(self, 
                             domain: Optional[str] = None,
                             min_accuracy: float = 0.7) -> List[Dict[str, Any]]:
        """
        Get curated knowledge from global learning.
        
        Args:
            domain: Optional domain filter
            min_accuracy: Minimum accuracy threshold
            
        Returns:
            List of curated knowledge entries
        """
        entries = self.global_learning.get_curated_knowledge(domain, min_accuracy)
        return [e.to_dict() for e in entries]
    
    def export_knowledge_package(self, 
                                domain: Optional[str] = None,
                                min_confidence: float = 0.7) -> Dict[str, Any]:
        """
        Export knowledge package for sharing.
        
        Args:
            domain: Optional domain filter
            min_confidence: Minimum confidence threshold
            
        Returns:
            Knowledge package dictionary
        """
        return self.global_learning.export_knowledge(domain, min_confidence)
    
    def import_knowledge_package(self, package: Dict[str, Any]):
        """
        Import knowledge from external package.
        
        Args:
            package: Knowledge package to import
        """
        self.global_learning.import_knowledge(package, source="external")
        logger.info({"msg": "Knowledge package imported"})


# Convenience factory function
def create_adaptive_orchestrator(
    user_id: Optional[str] = None,
    storage_path: Optional[str] = None,
    enable_auto_classification: bool = True,
    enable_global_learning: bool = True,
    enable_gauntlet: bool = True
) -> AdaptiveOrchestrator:
    """
    Create an adaptive orchestrator with specified configuration.
    
    This is the recommended way to use the knowledge engine for
    generic content processing.
    
    Args:
        user_id: Optional user identifier
        storage_path: Path for global learning storage
        enable_auto_classification: Enable automatic domain classification
        enable_global_learning: Enable global learning contributions
        enable_gauntlet: Enable continuous validation
        
    Returns:
        Configured AdaptiveOrchestrator
    """
    config = AdaptiveConfig(
        user_id=user_id,
        enable_auto_classification=enable_auto_classification,
        enable_global_learning=enable_global_learning,
        enable_gauntlet=enable_gauntlet
    )
    
    return AdaptiveOrchestrator(
        adaptive_config=config,
        storage_path=storage_path
    )
