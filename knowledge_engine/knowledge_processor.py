"""
Enhanced Knowledge Processing Pipeline for OpenEvolve Knowledge Engine

This module implements an advanced knowledge processing pipeline that provides:
- Knowledge artifact transformation and normalization
- Semantic enrichment and enhancement
- Quality assessment and validation
- Relationship mapping and entity linking
- Contextual analysis and categorization
- Multi-stage processing with error handling
"""

import json
import logging
import hashlib
import re
from typing import Dict, Any, List, Optional, Tuple, Set
from datetime import datetime
from collections import defaultdict
import statistics

# Import knowledge artifacts and other components
try:
    from .knowledge_extractor import KnowledgeArtifact
    from .knowledge_storage import KnowledgeStorage
except ImportError:
    from knowledge_extractor import KnowledgeArtifact
    from knowledge_storage import KnowledgeStorage

# Configure logging
logger = logging.getLogger(__name__)

class KnowledgeProcessor:
    """
    Advanced Knowledge Processor for OpenEvolve Knowledge Engine.
    
    This class implements a comprehensive knowledge processing pipeline with:
    - Multi-stage processing workflow
    - Semantic enrichment and transformation
    - Quality validation and assessment
    - Entity recognition and relationship mapping
    - Contextual analysis and categorization
    - Performance optimization and caching
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the knowledge processor.
        
        Args:
            config: Configuration dictionary with processing parameters
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Processing pipeline configuration
        self.processing_stages = [
            'normalization',
            'enrichment',
            'validation',
            'relationship_mapping',
            'quality_assessment',
            'contextual_analysis'
        ]
        
        # Quality thresholds
        self.quality_thresholds = {
            'excellent': 0.90,
            'good': 0.75,
            'fair': 0.50,
            'poor': 0.30
        }
        
        # Processing statistics
        self.processed_artifacts = 0
        self.enhanced_artifacts = 0
        self.filtered_artifacts = 0
        self.processing_times = []
        self.quality_distribution = defaultdict(int)
        
        # Knowledge enhancement patterns
        self.enhancement_patterns = self._initialize_enhancement_patterns()
        
        # Entity recognition patterns
        self.entity_patterns = self._initialize_entity_patterns()
        
        # Contextual categorization rules
        self.categorization_rules = self._initialize_categorization_rules()
        
        self.logger.info("Knowledge processor initialized with advanced processing pipeline")
    
    def _initialize_enhancement_patterns(self) -> Dict[str, Any]:
        """Initialize knowledge enhancement patterns"""
        return {
            'solution_patterns': {
                'hierarchical_decomposition': {
                    'enhancements': [
                        'add_hierarchical_structure_analysis',
                        'add_complexity_metrics',
                        'add_decomposition_strategy'
                    ],
                    'related_concepts': ['recursive_problem_solving', 'modular_design', 'abstraction']
                },
                'divide_conquer': {
                    'enhancements': [
                        'add_partitioning_strategy',
                        'add_combination_analysis',
                        'add_subproblem_relationships'
                    ],
                    'related_concepts': ['problem_decomposition', 'recursive_algorithm', 'parallel_processing']
                }
            },
            'critique_patterns': {
                'resource_allocation': {
                    'enhancements': [
                        'add_resource_utilization_metrics',
                        'add_allocation_strategy_analysis',
                        'add_optimization_recommendations'
                    ],
                    'related_concepts': ['load_balancing', 'workload_distribution', 'performance_optimization']
                },
                'algorithm_selection': {
                    'enhancements': [
                        'add_algorithm_comparison',
                        'add_complexity_analysis',
                        'add_adaptive_strategy'
                    ],
                    'related_concepts': ['computational_complexity', 'performance_benchmarking', 'adaptive_systems']
                }
            }
        }
    
    def _initialize_entity_patterns(self) -> Dict[str, Any]:
        """Initialize entity recognition patterns"""
        return {
            'domains': {
                'mathematical_optimization': ['optimization', 'gradient', 'constraint', 'objective_function'],
                'algebra': ['equation', 'variable', 'solver', 'linear_system'],
                'machine_learning': ['model', 'training', 'prediction', 'feature']
            },
            'problem_types': {
                'decomposition': ['decompose', 'partition', 'subproblem', 'hierarchy'],
                'search': ['search', 'exploration', 'traversal', 'heuristic'],
                'validation': ['validation', 'verification', 'testing', 'quality']
            },
            'methods': {
                'iterative': ['iteration', 'convergence', 'loop', 'recursive'],
                'analytical': ['analysis', 'theorem', 'proof', 'derivation'],
                'empirical': ['experiment', 'observation', 'measurement', 'testing']
            }
        }
    
    def _initialize_categorization_rules(self) -> Dict[str, Any]:
        """Initialize contextual categorization rules"""
        return {
            'complexity_levels': {
                'low': lambda artifact: artifact.content.get('complexity', 0) <= 3,
                'medium': lambda artifact: 4 <= artifact.content.get('complexity', 0) <= 6,
                'high': lambda artifact: artifact.content.get('complexity', 0) >= 7
            },
            'maturity_levels': {
                'experimental': lambda artifact: artifact.effectiveness_score < 0.6,
                'operational': lambda artifact: 0.6 <= artifact.effectiveness_score < 0.8,
                'optimized': lambda artifact: artifact.effectiveness_score >= 0.8
            },
            'applicability_scopes': {
                'narrow': lambda artifact: len(artifact.content.get('related_entities', [])) <= 2,
                'moderate': lambda artifact: 3 <= len(artifact.content.get('related_entities', [])) <= 5,
                'broad': lambda artifact: len(artifact.content.get('related_entities', [])) >= 6
            }
        }
    
    def process_knowledge_artifacts(self, artifacts: List[KnowledgeArtifact]) -> List[KnowledgeArtifact]:
        """
        Process knowledge artifacts through the complete enhancement pipeline.
        
        Args:
            artifacts: List of knowledge artifacts to process
            
        Returns:
            List of processed and enhanced knowledge artifacts
        """
        start_time = datetime.now()
        self.logger.info(f"Starting knowledge processing for {len(artifacts)} artifacts")
        
        processed_artifacts = []
        
        for artifact in artifacts:
            try:
                # Run through processing pipeline
                processed_artifact = self._run_processing_pipeline(artifact)
                
                if processed_artifact:
                    processed_artifacts.append(processed_artifact)
                    
                    # Update statistics
                    quality_score = processed_artifact.calculate_quality_score()
                    if quality_score >= self.quality_thresholds['excellent']:
                        self.quality_distribution['excellent'] += 1
                    elif quality_score >= self.quality_thresholds['good']:
                        self.quality_distribution['good'] += 1
                    elif quality_score >= self.quality_thresholds['fair']:
                        self.quality_distribution['fair'] += 1
                    else:
                        self.quality_distribution['poor'] += 1
                    
                    self.processed_artifacts += 1
                    if processed_artifact != artifact:
                        self.enhanced_artifacts += 1
                else:
                    self.filtered_artifacts += 1
                    
            except Exception as e:
                self.logger.error(f"Error processing artifact {artifact.id}: {e}")
                continue
        
        # Update processing time statistics
        processing_time = (datetime.now() - start_time).total_seconds()
        self.processing_times.append(processing_time)
        
        self.logger.info(f"Completed knowledge processing: {self.processed_artifacts} processed, " \
                        f"{self.enhanced_artifacts} enhanced, {self.filtered_artifacts} filtered")
        self.logger.info(f"Processing time: {processing_time:.3f}s")
        
        return processed_artifacts
    
    def _run_processing_pipeline(self, artifact: KnowledgeArtifact) -> Optional[KnowledgeArtifact]:
        """Run artifact through the complete processing pipeline"""
        try:
            # Stage 1: Normalization
            normalized_artifact = self._normalize_artifact(artifact)
            
            # Stage 2: Semantic Enrichment
            enriched_artifact = self._enrich_artifact_semantically(normalized_artifact)
            
            # Stage 3: Validation
            validation_result = self._validate_artifact(enriched_artifact)
            if not validation_result:
                self.logger.debug(f"Artifact {artifact.id} failed validation")
                return None
            
            # Stage 4: Relationship Mapping
            mapped_artifact = self._map_relationships(enriched_artifact)
            
            # Stage 5: Quality Assessment
            assessed_artifact = self._assess_quality(mapped_artifact)
            
            # Stage 6: Contextual Analysis
            final_artifact = self._perform_contextual_analysis(assessed_artifact)
            
            return final_artifact
            
        except Exception as e:
            self.logger.error(f"Pipeline processing failed for artifact {artifact.id}: {e}")
            return None
    
    def _normalize_artifact(self, artifact: KnowledgeArtifact) -> KnowledgeArtifact:
        """Normalize artifact structure and content"""
        # Create a copy to avoid modifying original
        normalized = KnowledgeArtifact(**artifact.to_dict())
        
        # Standardize field names and formats
        if 'content' in normalized.content:
            content = normalized.content['content']
            if isinstance(content, str) and len(content) > 1000:
                normalized.content['content_summary'] = content[:200] + '...'
        
        # Ensure required metadata fields
        if 'processing_metadata' not in normalized.metadata:
            normalized.metadata['processing_metadata'] = {}
        
        normalized.metadata['processing_metadata']['normalization'] = {
            'timestamp': datetime.now().isoformat(),
            'status': 'completed'
        }
        
        # Update version and timestamp
        normalized.last_updated = datetime.now().timestamp()
        normalized.version = f"{float(normalized.version) + 0.1:.1f}"
        
        return normalized
    
    def _enrich_artifact_semantically(self, artifact: KnowledgeArtifact) -> KnowledgeArtifact:
        """Enhance artifact with semantic information and additional context"""
        enriched = KnowledgeArtifact(**artifact.to_dict())
        
        # Pattern-based enrichment
        pattern_enhancements = self._apply_pattern_enhancements(enriched)
        
        # Entity recognition and linking
        entities = self._recognize_entities(enriched)
        if entities:
            if 'related_entities' not in enriched.content:
                enriched.content['related_entities'] = []
            enriched.content['related_entities'].extend(list(entities))
            
            # Update metadata
            enriched.metadata['recognized_entities'] = list(entities)
        
        # Add semantic relationships
        semantic_relationships = self._identify_semantic_relationships(enriched)
        if semantic_relationships:
            enriched.metadata['semantic_relationships'] = semantic_relationships
        
        # Update processing metadata
        enriched.metadata['processing_metadata']['enrichment'] = {
            'timestamp': datetime.now().isoformat(),
            'entities_discovered': len(entities) if entities else 0,
            'relationships_identified': len(semantic_relationships) if semantic_relationships else 0,
            'pattern_enhancements': list(pattern_enhancements.keys())
        }
        
        # Update version and quality indicators
        enriched.source_quality = min(1.0, enriched.source_quality + 0.05)
        enriched.last_updated = datetime.now().timestamp()
        enriched.version = f"{float(enriched.version) + 0.1:.1f}"
        
        return enriched
    
    def _apply_pattern_enhancements(self, artifact: KnowledgeArtifact) -> Dict[str, Any]:
        """Apply pattern-specific enhancements to the artifact"""
        enhancements_applied = {}
        
        # Check for solution pattern enhancements
        if artifact.artifact_type == 'solution_pattern':
            pattern_type = artifact.content.get('pattern_type', 'generic')
            if pattern_type in self.enhancement_patterns['solution_patterns']:
                pattern_data = self.enhancement_patterns['solution_patterns'][pattern_type]
                
                for enhancement in pattern_data['enhancements']:
                    if enhancement == 'add_hierarchical_structure_analysis':
                        self._add_hierarchical_analysis(artifact)
                        enhancements_applied[enhancement] = 'completed'
                    elif enhancement == 'add_complexity_metrics':
                        self._add_complexity_metrics(artifact)
                        enhancements_applied[enhancement] = 'completed'
        
        # Check for critique pattern enhancements
        elif artifact.artifact_type == 'critique_insight':
            pattern_type = artifact.content.get('pattern_type', 'generic')
            if pattern_type in self.enhancement_patterns['critique_patterns']:
                pattern_data = self.enhancement_patterns['critique_patterns'][pattern_type]
                
                for enhancement in pattern_data['enhancements']:
                    if enhancement == 'add_resource_utilization_metrics':
                        self._add_resource_metrics(artifact)
                        enhancements_applied[enhancement] = 'completed'
                    elif enhancement == 'add_algorithm_comparison':
                        self._add_algorithm_comparison(artifact)
                        enhancements_applied[enhancement] = 'completed'
        
        return enhancements_applied
    
    def _add_hierarchical_analysis(self, artifact: KnowledgeArtifact):
        """Add hierarchical structure analysis to solution patterns"""
        if 'hierarchical_analysis' not in artifact.content:
            artifact.content['hierarchical_analysis'] = {
                'structure_type': 'tree-based',
                'levels': artifact.content.get('complexity', 5),
                'modularity_score': min(1.0, artifact.content.get('complexity', 5) / 10.0),
                'abstraction_levels': ['high', 'medium', 'low']
            }
    
    def _add_complexity_metrics(self, artifact: KnowledgeArtifact):
        """Add complexity metrics to solution patterns"""
        if 'complexity_metrics' not in artifact.content:
            complexity = artifact.content.get('complexity', 5)
            artifact.content['complexity_metrics'] = {
                'computational_complexity': f'O(n^{complexity})',
                'cognitive_complexity': 'medium' if complexity <= 6 else 'high',
                'implementation_effort': complexity * 10,
                'maintenance_score': max(0.1, 1.0 - (complexity / 10.0))
            }
    
    def _add_resource_metrics(self, artifact: KnowledgeArtifact):
        """Add resource utilization metrics to critique insights"""
        if 'resource_metrics' not in artifact.content:
            artifact.content['resource_metrics'] = {
                'cpu_utilization': 0.75,
                'memory_usage': 'high',
                'io_operations': 'moderate',
                'network_usage': 'low'
            }
    
    def _add_algorithm_comparison(self, artifact: KnowledgeArtifact):
        """Add algorithm comparison analysis to critique insights"""
        if 'algorithm_comparison' not in artifact.content:
            artifact.content['algorithm_comparison'] = {
                'current_algorithm': artifact.content.get('solution_approach', 'unknown'),
                'alternative_algorithms': ['alternative_1', 'alternative_2'],
                'performance_comparison': {
                    'speed': 'current: medium, alternative_1: high, alternative_2: low',
                    'accuracy': 'current: high, alternative_1: medium, alternative_2: high',
                    'resource_usage': 'current: high, alternative_1: medium, alternative_2: low'
                }
            }
    
    def _recognize_entities(self, artifact: KnowledgeArtifact) -> Set[str]:
        """Recognize and extract entities from artifact content"""
        entities = set()
        
        # Extract text from various fields
        text_content = (
            f"{artifact.content.get('solution_approach', '')} " \
            f"{artifact.content.get('root_cause', '')} " \
            f"{artifact.content.get('prevention_strategy', '')} " \
            f"{artifact.content.get('team_name', '')} " \
            f"{artifact.content.get('pattern_type', '')}"
        ).lower()
        
        # Domain entity recognition
        for domain, keywords in self.entity_patterns['domains'].items():
            if any(keyword in text_content for keyword in keywords):
                entities.add(f"domain:{domain}")
        
        # Problem type recognition
        for problem_type, keywords in self.entity_patterns['problem_types'].items():
            if any(keyword in text_content for keyword in keywords):
                entities.add(f"problem_type:{problem_type}")
        
        # Method recognition
        for method, keywords in self.entity_patterns['methods'].items():
            if any(keyword in text_content for keyword in keywords):
                entities.add(f"method:{method}")
        
        # Add artifact-specific entities
        if artifact.domain:
            entities.add(f"domain:{artifact.domain.lower()}")
        
        if artifact.problem_type:
            entities.add(f"problem_type:{artifact.problem_type.lower()}")
        
        return entities
    
    def _identify_semantic_relationships(self, artifact: KnowledgeArtifact) -> List[Dict[str, Any]]:
        """Identify semantic relationships between entities"""
        relationships = []
        entities = artifact.content.get('related_entities', [])
        
        # Simple relationship identification based on co-occurrence
        if len(entities) >= 2:
            for i, entity1 in enumerate(entities):
                for entity2 in entities[i+1:]:
                    relationships.append({
                        'source_entity': entity1,
                        'target_entity': entity2,
                        'relationship_type': 'semantic_association',
                        'confidence': 0.7,
                        'context': artifact.artifact_type
                    })
        
        return relationships
    
    def _validate_artifact(self, artifact: KnowledgeArtifact) -> bool:
        """Validate artifact quality and completeness"""
        validation_result = True
        validation_issues = []
        
        # Check required fields
        required_fields = ['id', 'artifact_type', 'content', 'source_workflow_id']
        for field in required_fields:
            if not getattr(artifact, field, None):
                validation_issues.append(f"Missing required field: {field}")
                validation_result = False
        
        # Check content quality
        content = artifact.content
        if not content or len(content) < 3:
            validation_issues.append("Insufficient content")
            validation_result = False
        
        # Check quality metrics
        quality_score = artifact.calculate_quality_score()
        if quality_score < self.quality_thresholds['poor']:
            validation_issues.append(f"Low quality score: {quality_score:.2f}")
            validation_result = False
        
        # Update validation status
        artifact.validation_status = "validated" if validation_result else "invalid"
        artifact.confidence_score = 0.95 if validation_result else 0.3
        
        # Add validation metadata
        artifact.metadata['validation'] = {
            'status': artifact.validation_status,
            'timestamp': datetime.now().isoformat(),
            'issues': validation_issues,
            'quality_score': quality_score
        }
        
        return validation_result
    
    def _map_relationships(self, artifact: KnowledgeArtifact) -> KnowledgeArtifact:
        """Map relationships between this artifact and other knowledge elements"""
        mapped = KnowledgeArtifact(**artifact.to_dict())
        
        # Add relationship metadata
        relationships = {
            'related_artifacts': mapped.related_artifacts,
            'semantic_relationships': mapped.metadata.get('semantic_relationships', []),
            'entity_relationships': []
        }
        
        # Identify potential relationships based on content
        entities = mapped.content.get('related_entities', [])
        for entity in entities:
            relationships['entity_relationships'].append({
                'entity': entity,
                'relationship_type': 'content_related',
                'confidence': 0.8
            })
        
        mapped.metadata['relationships'] = relationships
        
        # Update processing metadata
        mapped.metadata['processing_metadata']['relationship_mapping'] = {
            'timestamp': datetime.now().isoformat(),
            'relationships_mapped': len(relationships['entity_relationships'])
        }
        
        return mapped
    
    def _assess_quality(self, artifact: KnowledgeArtifact) -> KnowledgeArtifact:
        """Perform comprehensive quality assessment"""
        assessed = KnowledgeArtifact(**artifact.to_dict())
        
        # Calculate detailed quality metrics
        quality_metrics = {
            'completeness': self._calculate_completeness(assessed),
            'consistency': self._calculate_consistency(assessed),
            'relevance': self._calculate_relevance(assessed),
            'accuracy': self._calculate_accuracy(assessed),
            'timeliness': self._calculate_timeliness(assessed)
        }
        
        # Calculate overall quality score
        overall_quality = self._calculate_overall_quality(quality_metrics)
        
        # Determine quality category
        if overall_quality >= self.quality_thresholds['excellent']:
            quality_category = 'excellent'
        elif overall_quality >= self.quality_thresholds['good']:
            quality_category = 'good'
        elif overall_quality >= self.quality_thresholds['fair']:
            quality_category = 'fair'
        else:
            quality_category = 'poor'
        
        # Update artifact quality attributes
        assessed.metadata['quality_assessment'] = {
            'overall_quality': overall_quality,
            'quality_category': quality_category,
            'quality_metrics': quality_metrics,
            'timestamp': datetime.now().isoformat()
        }
        
        # Adjust confidence based on quality
        assessed.confidence_score = min(1.0, assessed.confidence_score + (overall_quality * 0.1))
        
        # Update processing metadata
        assessed.metadata['processing_metadata']['quality_assessment'] = {
            'timestamp': datetime.now().isoformat(),
            'quality_category': quality_category,
            'quality_score': overall_quality
        }
        
        return assessed
    
    def _calculate_completeness(self, artifact: KnowledgeArtifact) -> float:
        """Calculate completeness score (0-1)"""
        required_fields = ['id', 'artifact_type', 'content', 'source_workflow_id', 'extraction_timestamp']
        present_fields = sum(1 for field in required_fields if getattr(artifact, field, None))
        
        content_fields = ['solution_approach', 'root_cause', 'prevention_strategy', 'team_name']
        content_present = sum(1 for field in content_fields if artifact.content.get(field))
        
        completeness = (present_fields / len(required_fields) * 0.6) + \
                      (content_present / len(content_fields) * 0.4) if content_fields else 0.6
        
        return completeness
    
    def _calculate_consistency(self, artifact: KnowledgeArtifact) -> float:
        """Calculate consistency score (0-1)"""
        consistency_score = 0.7  # Base score
        
        # Check if metadata aligns with content
        if artifact.domain and 'domain' in str(artifact.content).lower():
            consistency_score += 0.1
        
        if artifact.problem_type and 'problem' in str(artifact.content).lower():
            consistency_score += 0.1
        
        # Check if quality metrics are consistent
        quality_score = artifact.calculate_quality_score()
        if quality_score > 0.7 and artifact.confidence_score > 0.7:
            consistency_score += 0.1
        
        return min(consistency_score, 1.0)
    
    def _calculate_relevance(self, artifact: KnowledgeArtifact) -> float:
        """Calculate relevance score (0-1)"""
        relevance_score = 0.5
        
        # Content length factor
        content_text = str(artifact.content)
        if len(content_text) > 500:
            relevance_score += 0.2
        elif len(content_text) > 200:
            relevance_score += 0.1
        
        # Metadata presence factor
        if artifact.metadata and len(artifact.metadata) > 3:
            relevance_score += 0.2
        
        # Related entities factor
        entities = artifact.content.get('related_entities', [])
        if len(entities) >= 3:
            relevance_score += 0.2
        elif len(entities) >= 1:
            relevance_score += 0.1
        
        return min(relevance_score, 1.0)
    
    def _calculate_accuracy(self, artifact: KnowledgeArtifact) -> float:
        """Calculate accuracy score (0-1)"""
        accuracy_score = 0.7
        
        # Validation status factor
        if artifact.validation_status == 'validated':
            accuracy_score += 0.2
        elif artifact.validation_status == 'unvalidated':
            accuracy_score -= 0.1
        
        # Source quality factor
        if artifact.source_quality > 0.8:
            accuracy_score += 0.15
        elif artifact.source_quality > 0.6:
            accuracy_score += 0.05
        
        # Confidence factor
        if artifact.confidence_score > 0.8:
            accuracy_score += 0.1
        
        return min(accuracy_score, 1.0)
    
    def _calculate_timeliness(self, artifact: KnowledgeArtifact) -> float:
        """Calculate timeliness score (0-1)"""
        if not artifact.extraction_timestamp:
            return 0.5
        
        # Calculate age in days
        age_days = (datetime.now().timestamp() - artifact.extraction_timestamp) / (24 * 3600)
        
        # Timeliness decreases with age
        timeliness = max(0.1, 1.0 - (age_days / 30.0))
        
        return timeliness
    
    def _calculate_overall_quality(self, metrics: Dict[str, float]) -> float:
        """Calculate overall quality score from individual metrics"""
        weights = {
            'completeness': 0.25,
            'consistency': 0.20,
            'relevance': 0.20,
            'accuracy': 0.25,
            'timeliness': 0.10
        }
        
        weighted_sum = sum(metrics.get(key, 0.0) * weight for key, weight in weights.items())
        total_weight = sum(weights.values())
        
        return weighted_sum / total_weight if total_weight > 0 else 0.0
    
    def _perform_contextual_analysis(self, artifact: KnowledgeArtifact) -> KnowledgeArtifact:
        """Perform contextual analysis and categorization"""
        analyzed = KnowledgeArtifact(**artifact.to_dict())
        
        # Categorize by complexity
        complexity_category = self._categorize_complexity(analyzed)
        
        # Categorize by maturity
        maturity_category = self._categorize_maturity(analyzed)
        
        # Categorize by applicability
        applicability_category = self._categorize_applicability(analyzed)
        
        # Add contextual metadata
        analyzed.metadata['contextual_analysis'] = {
            'complexity_category': complexity_category,
            'maturity_category': maturity_category,
            'applicability_category': applicability_category,
            'contextual_tags': self._generate_contextual_tags(analyzed),
            'timestamp': datetime.now().isoformat()
        }
        
        # Update artifact attributes based on analysis
        analyzed.applicability_scope = applicability_category
        
        # Update processing metadata
        analyzed.metadata['processing_metadata']['contextual_analysis'] = {
            'timestamp': datetime.now().isoformat(),
            'complexity': complexity_category,
            'maturity': maturity_category,
            'applicability': applicability_category
        }
        
        return analyzed
    
    def _categorize_complexity(self, artifact: KnowledgeArtifact) -> str:
        """Categorize artifact by complexity"""
        for category, rule in self.categorization_rules['complexity_levels'].items():
            if rule(artifact):
                return category
        return 'medium'
    
    def _categorize_maturity(self, artifact: KnowledgeArtifact) -> str:
        """Categorize artifact by maturity"""
        for category, rule in self.categorization_rules['maturity_levels'].items():
            if rule(artifact):
                return category
        return 'operational'
    
    def _categorize_applicability(self, artifact: KnowledgeArtifact) -> str:
        """Categorize artifact by applicability scope"""
        for category, rule in self.categorization_rules['applicability_scopes'].items():
            if rule(artifact):
                return category
        return 'moderate'
    
    def _generate_contextual_tags(self, artifact: KnowledgeArtifact) -> List[str]:
        """Generate contextual tags based on artifact analysis"""
        tags = []
        
        # Add domain tag
        if artifact.domain:
            tags.append(f"domain:{artifact.domain}")
        
        # Add problem type tag
        if artifact.problem_type:
            tags.append(f"problem:{artifact.problem_type}")
        
        # Add complexity tag
        complexity = artifact.content.get('complexity', 5)
        if complexity >= 7:
            tags.append('complexity:high')
        elif complexity <= 3:
            tags.append('complexity:low')
        else:
            tags.append('complexity:medium')
        
        # Add quality tag
        quality_score = artifact.calculate_quality_score()
        if quality_score >= 0.8:
            tags.append('quality:high')
        elif quality_score >= 0.6:
            tags.append('quality:medium')
        else:
            tags.append('quality:low')
        
        return tags
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get comprehensive processing statistics"""
        stats = {
            'total_processed': self.processed_artifacts,
            'total_enhanced': self.enhanced_artifacts,
            'total_filtered': self.filtered_artifacts,
            'enhancement_rate': (self.enhanced_artifacts / self.processed_artifacts) if self.processed_artifacts > 0 else 0.0,
            'filter_rate': (self.filtered_artifacts / (self.processed_artifacts + self.filtered_artifacts)) if (self.processed_artifacts + self.filtered_artifacts) > 0 else 0.0,
            'quality_distribution': dict(self.quality_distribution),
            'average_processing_time': statistics.mean(self.processing_times) if self.processing_times else 0.0,
            'total_processing_time': sum(self.processing_times)
        }
        
        # Calculate percentages
        total_artifacts = sum(stats['quality_distribution'].values())
        if total_artifacts > 0:
            for category in stats['quality_distribution']:
                stats[f'{category}_percentage'] = (stats['quality_distribution'][category] / total_artifacts) * 100
        
        return stats
    
    def reset_stats(self):
        """Reset processing statistics"""
        self.processed_artifacts = 0
        self.enhanced_artifacts = 0
        self.filtered_artifacts = 0
        self.processing_times = []
        self.quality_distribution = defaultdict(int)

# Example usage and testing
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Create knowledge processor
    processor = KnowledgeProcessor()
    
    # Create example knowledge artifacts (simplified for demonstration)
    example_artifacts = [
        KnowledgeArtifact(
            id='test_artifact_001',
            artifact_type='solution_pattern',
            content={
                'solution_id': 'sol_001',
                'problem_type': 'nonlinear_optimization',
                'solution_approach': 'hierarchical gradient descent with adaptive learning rate',
                'success_rate': 0.95,
                'complexity': 8,
                'pattern_type': 'hierarchical_decomposition'
            },
            source_workflow_id='workflow_001',
            extraction_timestamp=datetime.now().timestamp(),
            domain='mathematical_optimization',
            problem_type='nonlinear_optimization',
            effectiveness_score=0.95
        ),
        KnowledgeArtifact(
            id='test_artifact_002',
            artifact_type='critique_insight',
            content={
                'critique_id': 'crit_001',
                'issue_type': 'resource allocation inefficiency',
                'root_cause': 'suboptimal workload distribution across processing units',
                'prevention_strategy': 'implement dynamic resource allocation with load balancing',
                'severity': 'high',
                'pattern_type': 'resource_allocation'
            },
            source_workflow_id='workflow_001',
            extraction_timestamp=datetime.now().timestamp(),
            domain='mathematical_optimization',
            problem_type='performance_optimization',
            effectiveness_score=0.85
        )
    ]
    
    print("Starting knowledge processing...")
    
    # Process artifacts
    processed_artifacts = processor.process_knowledge_artifacts(example_artifacts)
    
    print(f"\nProcessed {len(processed_artifacts)} artifacts:")
    for i, artifact in enumerate(processed_artifacts, 1):
        print(f"\nArtifact {i}: {artifact.artifact_type} - {artifact.id}")
        print(f"  - Version: {artifact.version}")
        print(f"  - Quality: {artifact.calculate_quality_score():.2f}")
        print(f"  - Confidence: {artifact.confidence_score:.2f}")
        print(f"  - Validation: {artifact.validation_status}")
        
        # Show processing metadata
        if 'processing_metadata' in artifact.metadata:
            print(f"  - Processing stages: {list(artifact.metadata['processing_metadata'].keys())}")
        
        # Show quality assessment
        if 'quality_assessment' in artifact.metadata:
            quality = artifact.metadata['quality_assessment']
            print(f"  - Quality category: {quality['quality_category']}")
            print(f"  - Quality score: {quality['overall_quality']:.2f}")
        
        # Show contextual analysis
        if 'contextual_analysis' in artifact.metadata:
            context = artifact.metadata['contextual_analysis']
            print(f"  - Contextual tags: {context['contextual_tags']}")
    
    # Get processing statistics
    stats = processor.get_processing_stats()
    print(f"\nProcessing Statistics:")
    print(f"  - Total processed: {stats['total_processed']}")
    print(f"  - Total enhanced: {stats['total_enhanced']}")
    print(f"  - Enhancement rate: {stats['enhancement_rate']:.2f}")
    print(f"  - Quality distribution: {stats['quality_distribution']}")
    print(f"  - Average processing time: {stats['average_processing_time']:.3f}s")
    
    print(f"\nKnowledge processing completed successfully!")