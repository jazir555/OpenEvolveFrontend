"""
Knowledge Validation and Quality Assurance System for OpenEvolve Knowledge Engine

This module implements a comprehensive validation and quality assurance framework that provides:
- Automated knowledge artifact validation
- Quality assurance workflows
- Rule-based validation engines
- Continuous quality monitoring
- Validation reporting and analytics
- Integration with processing pipeline
"""

import json
import logging
import hashlib
import re
from typing import Dict, Any, List, Optional, Tuple, Set, Callable
from datetime import datetime
from collections import defaultdict
import statistics

# Import knowledge artifacts and other components
try:
    from .knowledge_extractor import KnowledgeArtifact
    from .knowledge_processor import KnowledgeProcessor
except ImportError:
    from knowledge_extractor import KnowledgeArtifact
    from knowledge_processor import KnowledgeProcessor

# **LEAN INTEGRATION**: Formal verification with Lean
try:
    from leanaide_client import LeanAideClient
    LEAN_AVAILABLE = True
except ImportError:
    LEAN_AVAILABLE = False

# Configure logging
logger = logging.getLogger(__name__)

class KnowledgeValidator:
    """
    Advanced Knowledge Validator for OpenEvolve Knowledge Engine.
    
    This class implements a comprehensive validation and quality assurance system with:
    - Multi-level validation rules
    - Automated validation workflows
    - Quality assurance monitoring
    - Validation reporting and analytics
    - Continuous improvement tracking
    - Integration with knowledge processing pipeline
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the knowledge validator.
        
        Args:
            config: Configuration dictionary with validation parameters
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Validation rule sets
        self.validation_rules = self._initialize_validation_rules()
        self.quality_rules = self._initialize_quality_rules()
        self.compliance_rules = self._initialize_compliance_rules()
        
        # Validation statistics
        self.total_validations = 0
        self.successful_validations = 0
        self.failed_validations = 0
        self.validation_times = []
        self.quality_trends = defaultdict(list)
        self.compliance_metrics = defaultdict(int)
        
        # Quality thresholds
        self.quality_thresholds = {
            'excellent': 0.90,
            'good': 0.75,
            'fair': 0.50,
            'poor': 0.30,
            'minimum': 0.40
        }
        
        # Validation history
        self.validation_history = []
        
        self.logger.info("Knowledge validator initialized with comprehensive validation framework")
    
    def _initialize_validation_rules(self) -> Dict[str, Any]:
        """Initialize validation rules for different artifact types"""
        return {
            'knowledge_artifact': {
                'required_fields': ['id', 'artifact_type', 'content', 'source_workflow_id', 'extraction_timestamp'],
                'field_validators': {
                    'id': lambda x: isinstance(x, str) and len(x) > 5,
                    'artifact_type': lambda x: isinstance(x, str) and x in ['solution_pattern', 'critique_insight', 'team_performance', 'gauntlet_effectiveness', 'cross_cutting_analysis'],
                    'content': lambda x: isinstance(x, dict) and len(x) >= 3,
                    'source_workflow_id': lambda x: isinstance(x, str) and len(x) > 3,
                    'extraction_timestamp': lambda x: isinstance(x, (int, float)) and x > 0
                },
                'content_validators': {
                    'solution_pattern': {
                        'required_content_fields': ['solution_id', 'problem_type', 'solution_approach'],
                        'field_validators': {
                            'success_rate': lambda x: 0.0 <= x <= 1.0,
                            'complexity': lambda x: 1 <= x <= 10
                        }
                    },
                    'critique_insight': {
                        'required_content_fields': ['critique_id', 'issue_type', 'root_cause'],
                        'field_validators': {
                            'severity': lambda x: x in ['low', 'medium', 'high'],
                            'impact_score': lambda x: 0.0 <= x <= 1.0
                        }
                    }
                }
            },
            'quality_metrics': {
                'completeness': lambda x: 0.0 <= x <= 1.0,
                'consistency': lambda x: 0.0 <= x <= 1.0,
                'relevance': lambda x: 0.0 <= x <= 1.0,
                'accuracy': lambda x: 0.0 <= x <= 1.0,
                'timeliness': lambda x: 0.0 <= x <= 1.0,
                'overall_quality': lambda x: 0.0 <= x <= 1.0
            }
        }
    
    def _initialize_quality_rules(self) -> Dict[str, Any]:
        """Initialize quality assessment rules"""
        return {
            'quality_categories': {
                'excellent': {'min_score': 0.90, 'weight': 1.2},
                'good': {'min_score': 0.75, 'weight': 1.0},
                'fair': {'min_score': 0.50, 'weight': 0.8},
                'poor': {'min_score': 0.30, 'weight': 0.5},
                'invalid': {'min_score': 0.0, 'weight': 0.2}
            },
            'quality_improvement': {
                'excellent_to_excellent': {'threshold': 0.95, 'improvement': 'maintain'},
                'good_to_excellent': {'threshold': 0.85, 'improvement': 'enhance'},
                'fair_to_good': {'threshold': 0.65, 'improvement': 'improve'},
                'poor_to_fair': {'threshold': 0.45, 'improvement': 'remediate'}
            },
            'quality_degradation': {
                'excellent_to_good': {'threshold': 0.85, 'action': 'monitor'},
                'good_to_fair': {'threshold': 0.65, 'action': 'review'},
                'fair_to_poor': {'threshold': 0.45, 'action': 'remediate'}
            }
        }
    
    def _initialize_compliance_rules(self) -> Dict[str, Any]:
        """Initialize compliance and governance rules"""
        return {
            'data_quality': {
                'completeness': {'minimum': 0.8, 'target': 0.95},
                'accuracy': {'minimum': 0.85, 'target': 0.95},
                'consistency': {'minimum': 0.8, 'target': 0.95},
                'timeliness': {'minimum': 0.7, 'target': 0.9}
            },
            'metadata_standards': {
                'required_metadata': ['validation', 'quality_assessment', 'processing_metadata'],
                'validation_frequency': 'continuous',
                'quality_assessment_frequency': 'continuous'
            },
            'governance': {
                'validation_coverage': {'target': 100},
                'quality_improvement_rate': {'target': 5},
                'compliance_rate': {'target': 95}
            }
        }
    
    def validate_knowledge_artifacts(self, artifacts: List[KnowledgeArtifact]) -> Tuple[List[KnowledgeArtifact], Dict[str, Any]]:
        """
        Validate a list of knowledge artifacts using comprehensive validation rules.
        
        Args:
            artifacts: List of knowledge artifacts to validate
            
        Returns:
            Tuple of (valid_artifacts, validation_report)
        """
        start_time = datetime.now()
        self.logger.info(f"Starting validation of {len(artifacts)} knowledge artifacts")
        
        valid_artifacts = []
        validation_report = {
            'total_artifacts': len(artifacts),
            'valid_artifacts': 0,
            'invalid_artifacts': 0,
            'validation_details': [],
            'quality_distribution': defaultdict(int),
            'compliance_metrics': defaultdict(int)
        }
        
        for artifact in artifacts:
            try:
                # Perform comprehensive validation
                validation_result = self._validate_artifact_comprehensive(artifact)
                
                if validation_result['valid']:
                    valid_artifacts.append(artifact)
                    validation_report['valid_artifacts'] += 1
                    self.successful_validations += 1
                else:
                    validation_report['invalid_artifacts'] += 1
                    self.failed_validations += 1
                
                # Update validation statistics
                self.total_validations += 1
                validation_report['validation_details'].append(validation_result)
                
                # Track quality distribution
                quality_category = validation_result.get('quality_category', 'unknown')
                validation_report['quality_distribution'][quality_category] += 1
                
                # Update compliance metrics
                if validation_result['valid']:
                    validation_report['compliance_metrics']['valid_artifacts'] += 1
                else:
                    validation_report['compliance_metrics']['invalid_artifacts'] += 1
                
                # Track quality trends
                if validation_result.get('quality_score'):
                    self.quality_trends['overall'].append(validation_result['quality_score'])
                    self.quality_trends[quality_category].append(validation_result['quality_score'])
                
            except Exception as e:
                self.logger.error(f"Validation failed for artifact {artifact.id}: {str(e)}")
                validation_report['invalid_artifacts'] += 1
                self.failed_validations += 1
                self.total_validations += 1
                
                validation_report['validation_details'].append({
                    'artifact_id': artifact.id,
                    'valid': False,
                    'errors': [f"Validation exception: {str(e)}"],
                    'quality_score': 0.0,
                    'quality_category': 'invalid'
                })
        
        # Calculate validation time
        validation_time = (datetime.now() - start_time).total_seconds()
        self.validation_times.append(validation_time)
        
        # Generate validation summary
        validation_report['validation_time'] = validation_time
        validation_report['success_rate'] = validation_report['valid_artifacts'] / validation_report['total_artifacts'] if validation_report['total_artifacts'] > 0 else 0.0
        validation_report['compliance_rate'] = validation_report['compliance_metrics']['valid_artifacts'] / validation_report['total_artifacts'] if validation_report['total_artifacts'] > 0 else 0.0
        
        # Add validation to history
        self.validation_history.append({
            'timestamp': datetime.now().isoformat(),
            'report': validation_report
        })
        
        self.logger.info(f"Validation completed: {validation_report['valid_artifacts']} valid, " \
                        f"{validation_report['invalid_artifacts']} invalid, " \
                        f"time: {validation_time:.3f}s")
        
        return valid_artifacts, validation_report
    
    def _validate_artifact_comprehensive(self, artifact: KnowledgeArtifact) -> Dict[str, Any]:
        """Perform comprehensive validation of a knowledge artifact"""
        validation_result = {
            'artifact_id': artifact.id,
            'artifact_type': artifact.artifact_type,
            'valid': True,
            'errors': [],
            'warnings': [],
            'quality_score': 0.0,
            'quality_category': 'unknown',
            'compliance_status': 'compliant',
            'validation_timestamp': datetime.now().isoformat()
        }
        
        # Stage 1: Structural validation
        structural_validation = self._validate_structure(artifact)
        if not structural_validation['valid']:
            validation_result['valid'] = False
            validation_result['errors'].extend(structural_validation['errors'])
        
        # Stage 2: Content validation
        content_validation = self._validate_content(artifact)
        if not content_validation['valid']:
            validation_result['valid'] = False
            validation_result['errors'].extend(content_validation['errors'])
        
        # Stage 3: Quality validation
        quality_validation = self._validate_quality(artifact)
        validation_result.update(quality_validation)
        
        # Stage 4: Compliance validation
        compliance_validation = self._validate_compliance(artifact)
        validation_result.update(compliance_validation)
        
        # Determine overall validation status
        if validation_result['quality_score'] < self.quality_thresholds['minimum']:
            validation_result['valid'] = False
            validation_result['errors'].append(f"Quality score {validation_result['quality_score']:.2f} below minimum threshold {self.quality_thresholds['minimum']}")
        
        # Generate validation summary
        if validation_result['valid']:
            validation_result['validation_status'] = 'validated'
            validation_result['validation_level'] = 'full'
        else:
            validation_result['validation_status'] = 'invalid'
            validation_result['validation_level'] = 'partial' if validation_result['quality_score'] >= self.quality_thresholds['poor'] else 'failed'
        
        return validation_result
    
    def _validate_structure(self, artifact: KnowledgeArtifact) -> Dict[str, Any]:
        """Validate artifact structure and required fields"""
        validation = {'valid': True, 'errors': [], 'warnings': []}
        
        # Check required fields
        required_fields = self.validation_rules['knowledge_artifact']['required_fields']
        for field in required_fields:
            if not getattr(artifact, field, None):
                validation['valid'] = False
                validation['errors'].append(f"Missing required field: {field}")
            else:
                # Validate field format
                validator = self.validation_rules['knowledge_artifact']['field_validators'].get(field)
                if validator and not validator(getattr(artifact, field)):
                    validation['valid'] = False
                    validation['errors'].append(f"Invalid format for field: {field}")
        
        return validation
    
    def _validate_content(self, artifact: KnowledgeArtifact) -> Dict[str, Any]:
        """Validate artifact content based on artifact type"""
        validation = {'valid': True, 'errors': [], 'warnings': []}
        
        # Get content validators for artifact type
        content_validators = self.validation_rules['knowledge_artifact']['content_validators']
        artifact_validator = content_validators.get(artifact.artifact_type)
        
        if artifact_validator:
            # Check required content fields
            required_fields = artifact_validator.get('required_content_fields', [])
            for field in required_fields:
                if field not in artifact.content:
                    validation['valid'] = False
                    validation['errors'].append(f"Missing required content field: {field}")
            
            # Validate content field formats
            field_validators = artifact_validator.get('field_validators', {})
            for field, validator in field_validators.items():
                if field in artifact.content:
                    if not validator(artifact.content[field]):
                        validation['valid'] = False
                        validation['errors'].append(f"Invalid format for content field: {field}")
        
        # General content validation
        if not artifact.content or len(artifact.content) < 3:
            validation['valid'] = False
            validation['errors'].append("Insufficient content - minimum 3 fields required")
        
        return validation
    
    def _validate_quality(self, artifact: KnowledgeArtifact) -> Dict[str, Any]:
        """Validate artifact quality metrics"""
        quality_result = {
            'quality_score': 0.0,
            'quality_category': 'unknown',
            'quality_metrics': {},
            'quality_issues': []
        }
        
        # Calculate quality score if not already present
        if 'quality_assessment' in artifact.metadata:
            quality_result['quality_score'] = artifact.metadata['quality_assessment']['overall_quality']
            quality_result['quality_metrics'] = artifact.metadata['quality_assessment']['quality_metrics']
        else:
            # Calculate basic quality score
            quality_result['quality_score'] = artifact.calculate_quality_score()
            quality_result['quality_metrics'] = {
                'completeness': self._calculate_validation_completeness(artifact),
                'consistency': self._calculate_validation_consistency(artifact),
                'relevance': 0.7  # Default relevance for validation
            }
        
        # Validate quality metrics format
        for metric, validator in self.validation_rules['quality_metrics'].items():
            if metric in quality_result['quality_metrics']:
                if not validator(quality_result['quality_metrics'][metric]):
                    quality_result['quality_issues'].append(f"Invalid {metric} value: {quality_result['quality_metrics'][metric]}")
        
        # Determine quality category
        if quality_result['quality_score'] >= self.quality_thresholds['excellent']:
            quality_result['quality_category'] = 'excellent'
        elif quality_result['quality_score'] >= self.quality_thresholds['good']:
            quality_result['quality_category'] = 'good'
        elif quality_result['quality_score'] >= self.quality_thresholds['fair']:
            quality_result['quality_category'] = 'fair'
        elif quality_result['quality_score'] >= self.quality_thresholds['poor']:
            quality_result['quality_category'] = 'poor'
        else:
            quality_result['quality_category'] = 'invalid'
        
        return quality_result
    
    def _calculate_validation_completeness(self, artifact: KnowledgeArtifact) -> float:
        """Calculate completeness score for validation purposes"""
        required_fields = ['id', 'artifact_type', 'content', 'source_workflow_id', 'extraction_timestamp']
        present_fields = sum(1 for field in required_fields if getattr(artifact, field, None))
        
        content_fields = ['solution_approach', 'root_cause', 'prevention_strategy', 'team_name', 'pattern_type']
        content_present = sum(1 for field in content_fields if artifact.content.get(field))
        
        completeness = (present_fields / len(required_fields) * 0.7) + \
                      (content_present / len(content_fields) * 0.3) if content_fields else 0.7
        
        return completeness
    
    def _calculate_validation_consistency(self, artifact: KnowledgeArtifact) -> float:
        """Calculate consistency score for validation purposes"""
        consistency_score = 0.7
        
        # Check metadata consistency
        if artifact.domain and 'domain' in str(artifact.content).lower():
            consistency_score += 0.1
        
        if artifact.problem_type and 'problem' in str(artifact.content).lower():
            consistency_score += 0.1
        
        # Check quality consistency
        if hasattr(artifact, 'confidence_score') and artifact.confidence_score > 0.7:
            consistency_score += 0.1
        
        return min(consistency_score, 1.0)
    
    def _validate_compliance(self, artifact: KnowledgeArtifact) -> Dict[str, Any]:
        """Validate artifact compliance with governance rules"""
        compliance_result = {
            'compliance_status': 'compliant',
            'compliance_issues': [],
            'compliance_score': 1.0
        }
        
        # Check metadata standards compliance
        required_metadata = self.compliance_rules['metadata_standards']['required_metadata']
        for metadata_field in required_metadata:
            if metadata_field not in artifact.metadata:
                compliance_result['compliance_status'] = 'non_compliant'
                compliance_result['compliance_issues'].append(f"Missing required metadata: {metadata_field}")
                compliance_result['compliance_score'] -= 0.2
        
        # Check data quality compliance
        if 'quality_assessment' in artifact.metadata:
            quality_metrics = artifact.metadata['quality_assessment']['quality_metrics']
            
            # Check completeness compliance
            if quality_metrics.get('completeness', 0) < self.compliance_rules['data_quality']['completeness']['minimum']:
                compliance_result['compliance_status'] = 'non_compliant'
                compliance_result['compliance_issues'].append("Completeness below minimum threshold")
                compliance_result['compliance_score'] -= 0.15
            
            # Check accuracy compliance
            if quality_metrics.get('accuracy', 0) < self.compliance_rules['data_quality']['accuracy']['minimum']:
                compliance_result['compliance_status'] = 'non_compliant'
                compliance_result['compliance_issues'].append("Accuracy below minimum threshold")
                compliance_result['compliance_score'] -= 0.2
        
        compliance_result['compliance_score'] = max(0.0, compliance_result['compliance_score'])
        
        return compliance_result
    
    def assess_quality_improvement(self, before_validation: Dict[str, Any], 
                                  after_validation: Dict[str, Any]) -> Dict[str, Any]:
        """Assess quality improvement between validation cycles"""
        improvement_assessment = {
            'quality_improvement': 0.0,
            'category_change': 'none',
            'improvement_recommendations': []
        }
        
        before_score = before_validation.get('quality_score', 0.0)
        after_score = after_validation.get('quality_score', 0.0)
        
        improvement_assessment['quality_improvement'] = after_score - before_score
        
        # Determine category change
        before_category = before_validation.get('quality_category', 'unknown')
        after_category = after_validation.get('quality_category', 'unknown')
        
        if before_category != after_category:
            improvement_assessment['category_change'] = f"{before_category}_to_{after_category}"
            
            # Get improvement recommendations based on category change
            improvement_key = improvement_assessment['category_change']
            if improvement_key in self.quality_rules['quality_improvement']:
                improvement_assessment['improvement_recommendations'].append(
                    self.quality_rules['quality_improvement'][improvement_key]['improvement']
                )
        
        # Add general improvement recommendations
        if improvement_assessment['quality_improvement'] > 0.1:
            improvement_assessment['improvement_recommendations'].append("Continue current enhancement strategies")
        elif improvement_assessment['quality_improvement'] > 0.05:
            improvement_assessment['improvement_recommendations'].append("Maintain current quality levels")
        else:
            improvement_assessment['improvement_recommendations'].append("Review and enhance quality improvement processes")
        
        return improvement_assessment
    
    def monitor_quality_trends(self) -> Dict[str, Any]:
        """Monitor and analyze quality trends over time"""
        trend_analysis = {
            'overall_trend': 'stable',
            'category_trends': {},
            'quality_improvement_rate': 0.0,
            'compliance_trends': {}
        }
        
        # Analyze overall quality trend
        if len(self.quality_trends['overall']) >= 2:
            recent_scores = self.quality_trends['overall'][-5:]  # Last 5 validations
            if len(recent_scores) >= 2:
                avg_recent = statistics.mean(recent_scores[-2:])
                avg_historical = statistics.mean(recent_scores[:-2]) if len(recent_scores) > 2 else recent_scores[0]
                
                if avg_recent > avg_historical * 1.05:
                    trend_analysis['overall_trend'] = 'improving'
                elif avg_recent < avg_historical * 0.95:
                    trend_analysis['overall_trend'] = 'declining'
                else:
                    trend_analysis['overall_trend'] = 'stable'
                
                trend_analysis['quality_improvement_rate'] = ((avg_recent - avg_historical) / avg_historical) * 100 if avg_historical > 0 else 0.0
        
        # Analyze category trends
        for category in ['excellent', 'good', 'fair', 'poor']:
            if len(self.quality_trends[category]) >= 2:
                recent_category = len([s for s in self.quality_trends[category][-5:] if s >= self.quality_thresholds[category]])
                historical_category = len([s for s in self.quality_trends[category][:-5] if s >= self.quality_thresholds[category]]) if len(self.quality_trends[category]) > 5 else 0
                
                if recent_category > historical_category:
                    trend_analysis['category_trends'][category] = 'increasing'
                elif recent_category < historical_category:
                    trend_analysis['category_trends'][category] = 'decreasing'
                else:
                    trend_analysis['category_trends'][category] = 'stable'
        
        # Analyze compliance trends
        if self.total_validations >= 5:
            recent_compliance = self.compliance_metrics.get('valid_artifacts', 0) / min(5, self.total_validations)
            historical_compliance = (self.compliance_metrics.get('valid_artifacts', 0) - min(5, self.compliance_metrics.get('valid_artifacts', 0))) / max(1, self.total_validations - 5)
            
            if recent_compliance > historical_compliance * 1.05:
                trend_analysis['compliance_trends']['overall'] = 'improving'
            elif recent_compliance < historical_compliance * 0.95:
                trend_analysis['compliance_trends']['overall'] = 'declining'
            else:
                trend_analysis['compliance_trends']['overall'] = 'stable'
        
        return trend_analysis
    
    def generate_validation_report(self, validation_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate comprehensive validation report"""
        report = {
            'report_timestamp': datetime.now().isoformat(),
            'validation_summary': {
                'total_validations': len(validation_results),
                'successful_validations': sum(1 for r in validation_results if r['valid']),
                'failed_validations': sum(1 for r in validation_results if not r['valid']),
                'success_rate': sum(1 for r in validation_results if r['valid']) / len(validation_results) if validation_results else 0.0
            },
            'quality_distribution': defaultdict(int),
            'compliance_summary': {
                'compliant_artifacts': sum(1 for r in validation_results if r.get('compliance_status') == 'compliant'),
                'non_compliant_artifacts': sum(1 for r in validation_results if r.get('compliance_status') == 'non_compliant'),
                'compliance_rate': sum(1 for r in validation_results if r.get('compliance_status') == 'compliant') / len(validation_results) if validation_results else 0.0
            },
            'validation_details': validation_results,
            'recommendations': []
        }
        
        # Calculate quality distribution
        for result in validation_results:
            report['quality_distribution'][result.get('quality_category', 'unknown')] += 1
        
        # Generate recommendations
        if report['validation_summary']['success_rate'] >= 0.95:
            report['recommendations'].append("Excellent validation performance - maintain current standards")
        elif report['validation_summary']['success_rate'] >= 0.85:
            report['recommendations'].append("Good validation performance - consider targeted improvements")
        else:
            report['recommendations'].append("Review validation processes and quality standards")
        
        if report['compliance_summary']['compliance_rate'] >= 0.95:
            report['recommendations'].append("Excellent compliance - maintain governance standards")
        elif report['compliance_summary']['compliance_rate'] >= 0.85:
            report['recommendations'].append("Good compliance - address specific non-compliance issues")
        else:
            report['recommendations'].append("Review compliance requirements and governance processes")
        
        # Add quality improvement recommendations
        quality_distribution = report['quality_distribution']
        total = sum(quality_distribution.values())
        
        if total > 0:
            excellent_percentage = (quality_distribution.get('excellent', 0) / total) * 100
            poor_percentage = (quality_distribution.get('poor', 0) / total) * 100
            
            if excellent_percentage >= 70:
                report['recommendations'].append("High quality artifacts - focus on maintaining excellence")
            elif excellent_percentage >= 50:
                report['recommendations'].append("Good quality distribution - enhance fair/good artifacts to excellent")
            else:
                report['recommendations'].append("Quality improvement needed - implement enhancement strategies")
            
            if poor_percentage >= 20:
                report['recommendations'].append("High percentage of poor quality - implement remediation processes")
            elif poor_percentage >= 10:
                report['recommendations'].append("Address poor quality artifacts through targeted improvement")
        
        return report
    
    def get_validation_stats(self) -> Dict[str, Any]:
        """Get comprehensive validation statistics"""
        stats = {
            'total_validations': self.total_validations,
            'successful_validations': self.successful_validations,
            'failed_validations': self.failed_validations,
            'success_rate': self.successful_validations / self.total_validations if self.total_validations > 0 else 0.0,
            'failure_rate': self.failed_validations / self.total_validations if self.total_validations > 0 else 0.0,
            'average_validation_time': statistics.mean(self.validation_times) if self.validation_times else 0.0,
            'total_validation_time': sum(self.validation_times),
            'quality_trends': dict(self.quality_trends),
            'compliance_metrics': dict(self.compliance_metrics),
            'validation_history_count': len(self.validation_history)
        }
        
        # Calculate quality distribution percentages
        total_quality_samples = sum(len(scores) for scores in self.quality_trends.values())
        if total_quality_samples > 0:
            for category, scores in self.quality_trends.items():
                stats[f'{category}_percentage'] = (len(scores) / total_quality_samples) * 100
                stats[f'{category}_average'] = statistics.mean(scores) if scores else 0.0
        
        return stats
    
    def reset_stats(self):
        """Reset validation statistics"""
        self.total_validations = 0
        self.successful_validations = 0
        self.failed_validations = 0
        self.validation_times = []
        self.quality_trends = defaultdict(list)
        self.compliance_metrics = defaultdict(int)
        self.validation_history = []
    
    def get_validation_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent validation history"""
        return self.validation_history[-limit:] if limit else self.validation_history

    async def validate_with_lean(
        self,
        artifact: KnowledgeArtifact,
        criteria: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        **LEAN INTEGRATION**: Formal validation using Lean theorem prover.
        
        Performs mathematical formal validation of knowledge artifact content.
        
        Args:
            artifact: Knowledge artifact to validate
            criteria: Optional validation criteria
            
        Returns:
            Dict with formal validation results
        """
        if not LEAN_AVAILABLE:
            return {
                "verified": False,
                "reason": "Lean unavailable",
                "artifact_id": artifact.id if hasattr(artifact, 'id') else None,
                "formal_validation": False
            }
        
        try:
            logger.info(f"Running Lean formal validation for artifact {artifact.id}")
            
            client = LeanAideClient()
            content = str(artifact.content) if hasattr(artifact, 'content') else str(artifact)
            
            # Autoformalize the content
            formalized = await client.autoformalize(content)
            
            # Verify with Lean
            result = await client.verify(formalized)
            
            validation_result = {
                "verified": result.verified if hasattr(result, 'verified') else False,
                "confidence": result.confidence if hasattr(result, 'confidence') else 0.0,
                "proof": result.proof_code if hasattr(result, 'proof_code') else None,
                "artifact_id": artifact.id if hasattr(artifact, 'id') else None,
                "stored_in_knowledge_base": True,
                "verification_method": "lean_autoformalize",
                "formal_validation": True,
                "timestamp": datetime.now().isoformat()
            }
            
            # Update validation history
            self.validation_history.append({
                "timestamp": datetime.now().isoformat(),
                "artifact_id": artifact.id if hasattr(artifact, 'id') else None,
                "method": "lean",
                "result": validation_result
            })
            
            logger.info(f"Lean formal validation result: verified={validation_result['verified']}")
            return validation_result
            
        except Exception as e:
            logger.error(f"Lean formal validation error: {e}")
            return {
                "verified": False,
                "reason": str(e),
                "artifact_id": artifact.id if hasattr(artifact, 'id') else None,
                "formal_validation": False
            }

# Example usage and testing
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Create knowledge validator
    validator = KnowledgeValidator()
    
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
            effectiveness_score=0.95,
            metadata={
                'quality_assessment': {
                    'overall_quality': 0.92,
                    'quality_category': 'excellent',
                    'quality_metrics': {
                        'completeness': 0.95,
                        'consistency': 0.90,
                        'relevance': 0.90,
                        'accuracy': 0.95,
                        'timeliness': 0.85
                    }
                },
                'validation': {
                    'status': 'unvalidated',
                    'timestamp': datetime.now().isoformat()
                },
                'processing_metadata': {
                    'normalization': {'status': 'completed'},
                    'enrichment': {'status': 'completed'}
                }
            }
        ),
        KnowledgeArtifact(
            id='test_artifact_002',
            artifact_type='critique_insight',
            content={
                'critique_id': 'crit_001',
                'issue_type': 'resource allocation inefficiency',
                'root_cause': 'suboptimal workload distribution',
                'severity': 'high',
                'pattern_type': 'resource_allocation'
            },
            source_workflow_id='workflow_001',
            extraction_timestamp=datetime.now().timestamp(),
            domain='mathematical_optimization',
            problem_type='performance_optimization',
            effectiveness_score=0.75,
            metadata={
                'quality_assessment': {
                    'overall_quality': 0.78,
                    'quality_category': 'good',
                    'quality_metrics': {
                        'completeness': 0.85,
                        'consistency': 0.80,
                        'relevance': 0.75,
                        'accuracy': 0.80,
                        'timeliness': 0.70
                    }
                },
                'validation': {
                    'status': 'unvalidated',
                    'timestamp': datetime.now().isoformat()
                },
                'processing_metadata': {
                    'normalization': {'status': 'completed'},
                    'enrichment': {'status': 'completed'}
                }
            }
        ),
        KnowledgeArtifact(
            id='test_artifact_003',
            artifact_type='team_performance',
            content={
                'team_name': 'optimization_team',
                'success_rate': 0.92,
                'avg_response_time': 1.5
            },
            source_workflow_id='workflow_001',
            extraction_timestamp=datetime.now().timestamp(),
            domain='mathematical_optimization',
            problem_type='team_performance',
            effectiveness_score=0.55,
            metadata={
                'quality_assessment': {
                    'overall_quality': 0.58,
                    'quality_category': 'fair',
                    'quality_metrics': {
                        'completeness': 0.70,
                        'consistency': 0.65,
                        'relevance': 0.50,
                        'accuracy': 0.60,
                        'timeliness': 0.55
                    }
                },
                'validation': {
                    'status': 'unvalidated',
                    'timestamp': datetime.now().isoformat()
                },
                'processing_metadata': {
                    'normalization': {'status': 'completed'}
                }
            }
        )
    ]
    
    print("Starting knowledge validation...")
    
    # Validate artifacts
    valid_artifacts, validation_report = validator.validate_knowledge_artifacts(example_artifacts)
    
    print(f"\nValidation Results:")
    print(f"  - Total artifacts validated: {validation_report['total_artifacts']}")
    print(f"  - Valid artifacts: {validation_report['valid_artifacts']}")
    print(f"  - Invalid artifacts: {validation_report['invalid_artifacts']}")
    print(f"  - Success rate: {validation_report['success_rate']:.2f}")
    print(f"  - Compliance rate: {validation_report['compliance_rate']:.2f}")
    
    print(f"\nQuality Distribution:")
    for category, count in validation_report['quality_distribution'].items():
        print(f"  - {category}: {count}")
    
    print(f"\nValidation Details:")
    for i, detail in enumerate(validation_report['validation_details'], 1):
        print(f"\nArtifact {i}: {detail['artifact_id']}")
        print(f"  - Validation status: {detail['validation_status']}")
        print(f"  - Quality score: {detail['quality_score']:.2f}")
        print(f"  - Quality category: {detail['quality_category']}")
        print(f"  - Compliance status: {detail['compliance_status']}")
        if detail['errors']:
            print(f"  - Errors: {len(detail['errors'])} errors")
        if detail['warnings']:
            print(f"  - Warnings: {len(detail['warnings'])} warnings")
    
    # Generate comprehensive validation report
    comprehensive_report = validator.generate_validation_report(validation_report['validation_details'])
    print(f"\nComprehensive Validation Report:")
    print(f"  - Report timestamp: {comprehensive_report['report_timestamp']}")
    print(f"  - Validation success rate: {comprehensive_report['validation_summary']['success_rate']:.2f}")
    print(f"  - Compliance rate: {comprehensive_report['compliance_summary']['compliance_rate']:.2f}")
    print(f"  - Recommendations: {len(comprehensive_report['recommendations'])} recommendations")
    for i, recommendation in enumerate(comprehensive_report['recommendations'], 1):
        print(f"    {i}. {recommendation}")
    
    # Get validation statistics
    stats = validator.get_validation_stats()
    print(f"\nValidation Statistics:")
    print(f"  - Total validations: {stats['total_validations']}")
    print(f"  - Successful validations: {stats['successful_validations']}")
    print(f"  - Success rate: {stats['success_rate']:.2f}")
    print(f"  - Average validation time: {stats['average_validation_time']:.3f}s")
    
    # Monitor quality trends (simulated with multiple validations)
    print(f"\nSimulating quality trend analysis...")
    # Add some simulated historical data
    for _ in range(3):
        validator.quality_trends['excellent'].extend([0.91, 0.93])
        validator.quality_trends['good'].extend([0.78, 0.82])
        validator.total_validations += 2
        validator.successful_validations += 2
    
    trend_analysis = validator.monitor_quality_trends()
    print(f"  - Overall quality trend: {trend_analysis['overall_trend']}")
    print(f"  - Quality improvement rate: {trend_analysis['quality_improvement_rate']:.1f}%")
    print(f"  - Category trends: {trend_analysis['category_trends']}")
    
    print(f"\nKnowledge validation completed successfully!")