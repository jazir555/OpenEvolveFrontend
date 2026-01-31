"""
Knowledge Extractor for OpenEvolve Knowledge Engine

This module provides functionality for extracting knowledge artifacts from
workflow execution data, including solution patterns, critique patterns,
and team performance insights.
"""

import asyncio
import logging
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import uuid


logger = logging.getLogger(__name__)


@dataclass
class KnowledgeArtifact:
    """
    Representation of a knowledge artifact extracted from workflow execution.
    """
    artifact_id: str
    artifact_type: str
    content: str
    source: str
    context: str
    metadata: Dict[str, Any]
    timestamp: datetime
    effectiveness: Optional[float] = None
    confidence: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'artifact_id': self.artifact_id,
            'artifact_type': self.artifact_type,
            'content': self.content,
            'source': self.source,
            'context': self.context,
            'metadata': self.metadata,
            'timestamp': self.timestamp.isoformat(),
            'effectiveness': self.effectiveness,
            'confidence': self.confidence
        }


class KnowledgeExtractor:
    """
    Extractor for knowledge artifacts from workflow execution data.
    
    Provides methods for:
    - Extracting solution patterns
    - Identifying critique patterns
    - Analyzing team performance
    - Gauntlet effectiveness analysis
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the knowledge extractor.
        
        Args:
            config: Configuration for extraction
        """
        self.config = config or {}
        
        logger.info({
            "msg": "KnowledgeExtractor initialized",
            "config": self.config,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
    
    def extract_from_workflow(self, workflow_data: Dict[str, Any]) -> List[KnowledgeArtifact]:
        """
        Extract knowledge artifacts from workflow execution data.
        
        Args:
            workflow_data: Workflow execution data
            
        Returns:
            List of KnowledgeArtifact objects
        """
        start_time = datetime.now(timezone.utc)
        
        logger.info({
            "msg": "Starting knowledge extraction from workflow",
            "workflow_id": workflow_data.get('workflow_id'),
            "timestamp": start_time.isoformat()
        })
        
        artifacts = []
        
        # Extract solution patterns
        solution_patterns = workflow_data.get('solution_patterns', [])
        for pattern in solution_patterns:
            artifact = self._create_solution_pattern_artifact(pattern, workflow_data)
            if artifact:
                artifacts.append(artifact)
        
        # Extract critique patterns
        critique_patterns = workflow_data.get('critique_patterns', [])
        for pattern in critique_patterns:
            artifact = self._create_critique_pattern_artifact(pattern, workflow_data)
            if artifact:
                artifacts.append(artifact)
        
        # Extract team performance insights
        team_performance = workflow_data.get('team_performance', {})
        if team_performance:
            artifact = self._create_team_performance_artifact(team_performance, workflow_data)
            if artifact:
                artifacts.append(artifact)
        
        # Extract gauntlet effectiveness
        gauntlet_effectiveness = workflow_data.get('gauntlet_effectiveness', {})
        if gauntlet_effectiveness:
            artifact = self._create_gauntlet_effectiveness_artifact(gauntlet_effectiveness, workflow_data)
            if artifact:
                artifacts.append(artifact)
        
        # Extract general execution insights
        execution_insights = self._extract_execution_insights(workflow_data)
        for insight in execution_insights:
            artifact = self._create_general_artifact(insight, workflow_data)
            if artifact:
                artifacts.append(artifact)
        
        processing_time_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
        
        logger.info({
            "msg": "Knowledge extraction completed",
            "workflow_id": workflow_data.get('workflow_id'),
            "artifacts_extracted": len(artifacts),
            "processing_time_ms": processing_time_ms,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
        
        return artifacts
    
    def _create_solution_pattern_artifact(
        self,
        pattern: Dict[str, Any],
        workflow_data: Dict[str, Any]
    ) -> Optional[KnowledgeArtifact]:
        """Create a knowledge artifact from a solution pattern."""
        try:
            pattern_content = f"Solution Pattern: {pattern.get('pattern', 'Unknown')}\n"
            pattern_content += f"Effectiveness: {pattern.get('effectiveness', 'N/A')}\n"
            pattern_content += f"Context: {pattern.get('context', 'General')}\n"
            pattern_content += f"Description: {pattern.get('description', '')}"
            
            artifact = KnowledgeArtifact(
                artifact_id=f"sp_{uuid.uuid4().hex[:8]}",
                artifact_type="solution_pattern",
                content=pattern_content,
                source=workflow_data.get('workflow_id', 'unknown'),
                context=workflow_data.get('domain', 'general'),
                metadata={
                    'pattern_type': pattern.get('pattern', 'unknown'),
                    'effectiveness': pattern.get('effectiveness'),
                    'context': pattern.get('context', 'general'),
                    'complexity': workflow_data.get('complexity', 'unknown'),
                    'team_size': workflow_data.get('team_size', 1)
                },
                timestamp=datetime.fromisoformat(workflow_data.get('timestamp', datetime.now(timezone.utc).isoformat())),
                effectiveness=pattern.get('effectiveness'),
                confidence=self._calculate_confidence(pattern)
            )
            
            return artifact
        except Exception as e:
            logger.error({
                "msg": "Failed to create solution pattern artifact",
                "error": str(e),
                "pattern": pattern
            })
            return None
    
    def _create_critique_pattern_artifact(
        self,
        pattern: Dict[str, Any],
        workflow_data: Dict[str, Any]
    ) -> Optional[KnowledgeArtifact]:
        """Create a knowledge artifact from a critique pattern."""
        try:
            pattern_content = f"Critique Pattern: {pattern.get('pattern', 'Unknown')}\n"
            pattern_content += f"Issue: {pattern.get('issue', 'N/A')}\n"
            pattern_content += f"Severity: {pattern.get('severity', 'N/A')}\n"
            pattern_content += f"Description: {pattern.get('description', '')}"
            
            artifact = KnowledgeArtifact(
                artifact_id=f"cp_{uuid.uuid4().hex[:8]}",
                artifact_type="critique_pattern",
                content=pattern_content,
                source=workflow_data.get('workflow_id', 'unknown'),
                context=workflow_data.get('domain', 'general'),
                metadata={
                    'pattern_type': pattern.get('pattern', 'unknown'),
                    'issue': pattern.get('issue', 'unknown'),
                    'severity': pattern.get('severity', 'low'),
                    'complexity': workflow_data.get('complexity', 'unknown'),
                    'team_size': workflow_data.get('team_size', 1)
                },
                timestamp=datetime.fromisoformat(workflow_data.get('timestamp', datetime.now(timezone.utc).isoformat())),
                effectiveness=1.0 - self._severity_to_effectiveness(pattern.get('severity', 'low')),
                confidence=self._calculate_confidence(pattern)
            )
            
            return artifact
        except Exception as e:
            logger.error({
                "msg": "Failed to create critique pattern artifact",
                "error": str(e),
                "pattern": pattern
            })
            return None
    
    def _create_team_performance_artifact(
        self,
        performance: Dict[str, Any],
        workflow_data: Dict[str, Any]
    ) -> Optional[KnowledgeArtifact]:
        """Create a knowledge artifact from team performance data."""
        try:
            perf_content = f"Team Performance Report\n"
            perf_content += f"Efficiency: {performance.get('efficiency', 'N/A')}\n"
            perf_content += f"Collaboration: {performance.get('collaboration', 'N/A')}\n"
            perf_content += f"Adaptability: {performance.get('adaptability', 'N/A')}\n"
            perf_content += f"Team Size: {workflow_data.get('team_size', 1)}"
            
            avg_performance = (
                performance.get('efficiency', 0.5) + 
                performance.get('collaboration', 0.5) + 
                performance.get('adaptability', 0.5)
            ) / 3
            
            artifact = KnowledgeArtifact(
                artifact_id=f"tp_{uuid.uuid4().hex[:8]}",
                artifact_type="team_performance",
                content=perf_content,
                source=workflow_data.get('workflow_id', 'unknown'),
                context=workflow_data.get('domain', 'general'),
                metadata={
                    'efficiency': performance.get('efficiency'),
                    'collaboration': performance.get('collaboration'),
                    'adaptability': performance.get('adaptability'),
                    'team_size': workflow_data.get('team_size', 1),
                    'complexity': workflow_data.get('complexity', 'unknown')
                },
                timestamp=datetime.fromisoformat(workflow_data.get('timestamp', datetime.now(timezone.utc).isoformat())),
                effectiveness=avg_performance,
                confidence=0.8  # Team performance metrics tend to be reliable
            )
            
            return artifact
        except Exception as e:
            logger.error({
                "msg": "Failed to create team performance artifact",
                "error": str(e),
                "performance": performance
            })
            return None
    
    def _create_gauntlet_effectiveness_artifact(
        self,
        effectiveness: Dict[str, Any],
        workflow_data: Dict[str, Any]
    ) -> Optional[KnowledgeArtifact]:
        """Create a knowledge artifact from gauntlet effectiveness data."""
        try:
            eff_content = f"Gauntlet Effectiveness Report\n"
            eff_content += f"Completion Rate: {effectiveness.get('completion_rate', 'N/A')}\n"
            eff_content += f"Quality Score: {effectiveness.get('quality_score', 'N/A')}\n"
            eff_content += f"Iteration Count: {effectiveness.get('iteration_count', 'N/A')}\n"
            eff_content += f"Success: {workflow_data.get('success', 'N/A')}"
            
            overall_effectiveness = (
                effectiveness.get('completion_rate', 0.5) * 0.6 + 
                effectiveness.get('quality_score', 0.5) * 0.4
            )
            
            artifact = KnowledgeArtifact(
                artifact_id=f"ge_{uuid.uuid4().hex[:8]}",
                artifact_type="gauntlet_effectiveness",
                content=eff_content,
                source=workflow_data.get('workflow_id', 'unknown'),
                context=workflow_data.get('domain', 'general'),
                metadata={
                    'completion_rate': effectiveness.get('completion_rate'),
                    'quality_score': effectiveness.get('quality_score'),
                    'iteration_count': effectiveness.get('iteration_count'),
                    'success': workflow_data.get('success'),
                    'complexity': workflow_data.get('complexity', 'unknown')
                },
                timestamp=datetime.fromisoformat(workflow_data.get('timestamp', datetime.now(timezone.utc).isoformat())),
                effectiveness=overall_effectiveness,
                confidence=0.9  # Effectiveness metrics are usually reliable
            )
            
            return artifact
        except Exception as e:
            logger.error({
                "msg": "Failed to create gauntlet effectiveness artifact",
                "error": str(e),
                "effectiveness": effectiveness
            })
            return None
    
    def _extract_execution_insights(self, workflow_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract general insights from workflow execution."""
        insights = []
        
        # Execution time insights
        execution_time = workflow_data.get('execution_data', {}).get('execution_time')
        if execution_time:
            insights.append({
                'type': 'execution_time',
                'content': f"Execution took {execution_time} seconds",
                'metric': execution_time
            })
        
        # Success/failure insights
        success = workflow_data.get('success')
        if success is not None:
            insights.append({
                'type': 'outcome',
                'content': f"Workflow {'succeeded' if success else 'failed'}",
                'success': success
            })
        
        # Complexity insights
        complexity = workflow_data.get('complexity')
        if complexity:
            insights.append({
                'type': 'complexity',
                'content': f"Problem complexity: {complexity}",
                'complexity': complexity
            })
        
        return insights
    
    def _create_general_artifact(
        self,
        insight: Dict[str, Any],
        workflow_data: Dict[str, Any]
    ) -> Optional[KnowledgeArtifact]:
        """Create a general knowledge artifact from an insight."""
        try:
            artifact = KnowledgeArtifact(
                artifact_id=f"gi_{uuid.uuid4().hex[:8]}",
                artifact_type=insight['type'],
                content=insight['content'],
                source=workflow_data.get('workflow_id', 'unknown'),
                context=workflow_data.get('domain', 'general'),
                metadata={
                    'insight_type': insight['type'],
                    'complexity': workflow_data.get('complexity', 'unknown'),
                    'team_size': workflow_data.get('team_size', 1),
                    **{k: v for k, v in insight.items() if k not in ['type', 'content']}
                },
                timestamp=datetime.fromisoformat(workflow_data.get('timestamp', datetime.now(timezone.utc).isoformat())),
                effectiveness=insight.get('success', 0.5) if insight.get('type') == 'outcome' else 0.5,
                confidence=0.7
            )
            
            return artifact
        except Exception as e:
            logger.error({
                "msg": "Failed to create general artifact",
                "error": str(e),
                "insight": insight
            })
            return None
    
    def _calculate_confidence(self, data: Dict[str, Any]) -> float:
        """Calculate confidence score for an artifact."""
        # Base confidence on available data
        score = 0.5  # Base score
        
        # Increase for numeric values
        if 'effectiveness' in data and isinstance(data['effectiveness'], (int, float)):
            score += 0.2
        
        if 'severity' in data:
            score += 0.1
        
        # Cap at 1.0
        return min(score, 1.0)
    
    def _severity_to_effectiveness(self, severity: str) -> float:
        """Convert severity level to effectiveness (inverse relationship)."""
        severity_map = {
            'critical': 0.1,
            'high': 0.3,
            'medium': 0.5,
            'low': 0.8,
            'info': 0.9
        }
        return severity_map.get(severity.lower(), 0.5)
    
    def extract_quality_metrics(self, artifacts: List[KnowledgeArtifact]) -> Dict[str, Any]:
        """
        Extract quality metrics from a list of knowledge artifacts.
        
        Args:
            artifacts: List of KnowledgeArtifact objects
            
        Returns:
            Dictionary with quality metrics
        """
        if not artifacts:
            return {
                'total_artifacts': 0,
                'average_effectiveness': 0.0,
                'average_confidence': 0.0,
                'artifact_types': {},
                'time_span': None
            }
        
        total_artifacts = len(artifacts)
        avg_effectiveness = sum(a.effectiveness or 0 for a in artifacts) / total_artifacts
        avg_confidence = sum(a.confidence or 0 for a in artifacts) / total_artifacts
        
        # Count artifact types
        type_counts = {}
        for artifact in artifacts:
            art_type = artifact.artifact_type
            type_counts[art_type] = type_counts.get(art_type, 0) + 1
        
        # Calculate time span
        timestamps = [a.timestamp for a in artifacts]
        time_span = None
        if timestamps:
            time_span = {
                'start': min(timestamps).isoformat(),
                'end': max(timestamps).isoformat(),
                'duration_days': (max(timestamps) - min(timestamps)).days
            }
        
        return {
            'total_artifacts': total_artifacts,
            'average_effectiveness': avg_effectiveness,
            'average_confidence': avg_confidence,
            'artifact_types': type_counts,
            'time_span': time_span
        }