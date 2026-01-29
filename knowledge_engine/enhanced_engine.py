"""
Enhanced Knowledge Engine for OpenEvolve

This module provides the complete Phase 2 implementation of the OpenEvolve
Knowledge Engine with advanced features, performance optimization, and
machine learning integration as specified in the technical architecture.
"""

import asyncio
import json
import logging
import time
from datetime import datetime
from typing import Dict, Any, List, Optional

# Import existing components
try:
    from .core import KnowledgeState, EntityKnowledgeGraph
except ImportError:
    from core import KnowledgeState, EntityKnowledgeGraph

# Import Phase 1 components
try:
    from .knowledge_extractor import KnowledgeExtractor
except ImportError:
    from knowledge_extractor import KnowledgeExtractor

# Import Phase 2 components
try:
    from .enhanced_storage import EnhancedKnowledgeStorage
except ImportError:
    from enhanced_storage import EnhancedKnowledgeStorage

try:
    from .enhanced_retriever import EnhancedKnowledgeRetriever
except ImportError:
    from enhanced_retriever import EnhancedKnowledgeRetriever

try:
    from .embedding_generator import EmbeddingGenerator
except ImportError:
    from embedding_generator import EmbeddingGenerator

class EnhancedKnowledgeEngine:
    """
    Enhanced Knowledge Engine with Phase 2 capabilities
    
    This class provides a comprehensive knowledge management system with:
    - Advanced knowledge extraction and processing
    - Enhanced storage with performance optimization
    - Machine learning-based retrieval
    - Personalized recommendations
    - Comprehensive analytics and quality metrics
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the Enhanced Knowledge Engine.
        
        Args:
            config: Configuration dictionary for all components
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Initialize core components
        self.state = KnowledgeState()
        self.graph = EntityKnowledgeGraph()
        
        # Initialize Phase 1 components
        self.extractor = KnowledgeExtractor(self.config)
        
        # Initialize Phase 2 components
        self.storage = EnhancedKnowledgeStorage(self.config)
        self.retriever = EnhancedKnowledgeRetriever(self.storage, self.config)
        self.embedding_gen = EmbeddingGenerator(self.config)
        
        # Performance tracking
        self.performance_metrics = {
            'total_operations': 0,
            'average_processing_time': 0.0,
            'last_operation': None
        }
        
        self.logger.info("Enhanced Knowledge Engine initialized")
        
    def process_workflow_with_enhanced_features(self, workflow_data: Dict[str, Any],
                                               generate_embeddings: bool = True) -> Dict[str, Any]:
        """
        Process workflow data with enhanced features including embedding generation.
        
        Args:
            workflow_data: Workflow execution data
            generate_embeddings: Whether to generate embeddings for artifacts
            
        Returns:
            Processing results with enhanced features
        """
        start_time = time.time()
        
        try:
            # Extract knowledge artifacts
            artifacts = self.extractor.extract_from_workflow(workflow_data)
            
            # Generate embeddings for artifacts
            if generate_embeddings:
                for artifact in artifacts:
                    # Convert KnowledgeArtifact to dict for embedding generation
                    artifact_dict = {
                        'type': artifact.artifact_type,
                        'source': artifact.source,
                        'content': artifact.content,
                        'context': artifact.context,
                        'metadata': artifact.metadata
                    }
                    
                    # Generate embedding
                    embedding = self.embedding_gen.generate_knowledge_artifact_embedding(artifact_dict)
                    artifact_dict['embeddings'] = embedding
                    
                    # Store enhanced artifact
                    artifact_id = self.storage.store_knowledge_artifact(artifact_dict)
                    
                    # Update artifact with ID
                    artifact.artifact_id = artifact_id
            else:
                # Store without embeddings
                for artifact in artifacts:
                    artifact_dict = {
                        'type': artifact.artifact_type,
                        'source': artifact.source,
                        'content': artifact.content,
                        'context': artifact.context,
                        'metadata': artifact.metadata
                    }
                    
                    artifact_id = self.storage.store_knowledge_artifact(artifact_dict, generate_embedding=False)
                    artifact.artifact_id = artifact_id
            
            # Update knowledge state
            self.state.add_workflow_execution(
                workflow_id=workflow_data.get('workflow_id', 'unknown'),
                artifacts_extracted=len(artifacts),
                timestamp=workflow_data.get('timestamp')
            )
            
            # Update performance metrics
            processing_time = time.time() - start_time
            self._update_performance_metrics('workflow_processing', processing_time)
            
            return {
                'status': 'processed',
                'workflow_id': workflow_data.get('workflow_id', 'unknown'),
                'knowledge_extracted': len(artifacts),
                'artifacts_with_embeddings': len(artifacts) if generate_embeddings else 0,
                'processing_time': processing_time,
                'stored_artifacts': [
                    {
                        'artifact_id': artifact.artifact_id,
                        'type': artifact.artifact_type,
                        'source': artifact.source
                    }
                    for artifact in artifacts
                ]
            }
            
        except Exception as e:
            self.logger.error(f"Enhanced workflow processing failed: {str(e)}")
            return {
                'status': 'error',
                'error': str(e),
                'workflow_id': workflow_data.get('workflow_id', 'unknown')
            }
    
    def _update_performance_metrics(self, operation_type: str, execution_time: float):
        """Update performance metrics"""
        self.performance_metrics['total_operations'] += 1
        self.performance_metrics['last_operation'] = {
            'type': operation_type,
            'time': execution_time,
            'timestamp': datetime.now().isoformat()
        }
        
        # Update average processing time
        if self.performance_metrics['total_operations'] == 1:
            self.performance_metrics['average_processing_time'] = execution_time
        else:
            self.performance_metrics['average_processing_time'] = (
                self.performance_metrics['average_processing_time'] * 0.9 +
                execution_time * 0.1
            )
    
    def enhanced_search(self, query: str, query_type: str = 'hybrid',
                       filters: Optional[Dict[str, Any]] = None,
                       limit: int = 10, use_cache: bool = True) -> List[Dict[str, Any]]:
        """
        Perform enhanced search using the advanced retriever.
        
        Args:
            query: Search query string
            query_type: Type of search (hybrid, vector, keyword, semantic)
            filters: Additional filters
            limit: Maximum number of results
            use_cache: Whether to use caching
            
        Returns:
            List of search results
        """
        return self.retriever.search_knowledge(query, query_type, filters, limit, use_cache)
    
    def get_personalized_recommendations(self, context: Dict[str, Any],
                                         user_profile: Optional[Dict[str, Any]] = None,
                                         limit: int = 5) -> List[Dict[str, Any]]:
        """
        Get personalized recommendations using advanced algorithms.
        
        Args:
            context: Context dictionary
            user_profile: Optional user profile for personalization
            limit: Maximum number of recommendations
            
        Returns:
            List of personalized recommendations
        """
        return self.retriever.get_personalized_recommendations(context, user_profile, limit)
    
    def semantic_search(self, query: str, context: Optional[Dict[str, Any]] = None,
                       limit: int = 5) -> List[Dict[str, Any]]:
        """
        Perform semantic search using advanced embedding techniques.
        
        Args:
            query: Search query
            context: Optional context for semantic understanding
            limit: Maximum number of results
            
        Returns:
            List of semantically relevant results
        """
        return self.retriever.semantic_search(query, context, limit)
    
    def get_knowledge_analytics(self) -> Dict[str, Any]:
        """
        Get comprehensive analytics about the knowledge base.
        
        Returns:
            Dictionary containing detailed analytics
        """
        start_time = time.time()
        
        try:
            # Get storage statistics
            storage_stats = self.storage.get_aggregated_statistics()
            
            # Get quality metrics
            quality_metrics = self.retriever.get_knowledge_quality_metrics()
            
            # Get performance metrics
            retriever_performance = self.retriever.get_performance_metrics()
            
            # Get trend analysis
            trends = self.retriever.get_knowledge_trends(time_range='30d', analysis_type='advanced')
            
            # Combine all analytics
            analytics = {
                'storage_statistics': storage_stats,
                'quality_metrics': quality_metrics.get('quality_metrics', {}),
                'overall_quality_score': quality_metrics.get('overall_quality_score', 0),
                'trend_analysis': trends.get('trend_analysis', {}),
                'performance_metrics': {
                    'retriever': retriever_performance,
                    'engine': self.performance_metrics
                },
                'knowledge_graph': self.storage.create_knowledge_graph(),
                'generated_at': datetime.now().isoformat(),
                'analysis_time': time.time() - start_time
            }
            
            return analytics
            
        except Exception as e:
            self.logger.error(f"Failed to generate knowledge analytics: {str(e)}")
            return {'error': str(e)}
    
    def batch_process_workflows(self, workflows: List[Dict[str, Any]],
                               batch_size: int = 10) -> Dict[str, Any]:
        """
        Batch process multiple workflows with performance optimization.
        
        Args:
            workflows: List of workflow data
            batch_size: Batch size for processing
            
        Returns:
            Dictionary with batch processing results
        """
        start_time = time.time()
        
        results = {
            'total_workflows': len(workflows),
            'success_count': 0,
            'failed_count': 0,
            'artifacts_extracted': 0,
            'processing_times': [],
            'start_time': start_time,
            'end_time': None,
            'duration': None
        }
        
        try:
            # Process workflows in batches
            for i in range(0, len(workflows), batch_size):
                batch = workflows[i:i + batch_size]
                batch_results = []
                
                for workflow in batch:
                    try:
                        batch_start = time.time()
                        
                        # Process individual workflow
                        workflow_result = self.process_workflow_with_enhanced_features(workflow)
                        
                        if workflow_result['status'] == 'processed':
                            results['success_count'] += 1
                            results['artifacts_extracted'] += workflow_result['knowledge_extracted']
                        else:
                            results['failed_count'] += 1
                        
                        batch_results.append({
                            'workflow_id': workflow.get('workflow_id', f'batch_{i}_{len(batch_results)}'),
                            'status': workflow_result['status'],
                            'artifacts': workflow_result['knowledge_extracted']
                        })
                        
                        results['processing_times'].append(time.time() - batch_start)
                        
                    except Exception as e:
                        self.logger.error(f"Failed to process workflow in batch: {str(e)}")
                        results['failed_count'] += 1
                        batch_results.append({
                            'workflow_id': workflow.get('workflow_id', f'batch_{i}_{len(batch_results)}'),
                            'status': 'error',
                            'error': str(e)
                        })
                
                self.logger.info(f"Processed batch {i//batch_size + 1}: {len(batch_results)} workflows")
            
            results['end_time'] = time.time()
            results['duration'] = results['end_time'] - start_time
            results['average_processing_time'] = sum(results['processing_times']) / len(results['processing_times']) if results['processing_times'] else 0
            
            self.logger.info(f"Batch processing completed: {results['success_count']}/{results['total_workflows']} workflows")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Batch processing failed: {str(e)}")
            results['error'] = str(e)
            results['end_time'] = time.time()
            results['duration'] = results['end_time'] - start_time
            return results
    
    def optimize_knowledge_base(self) -> Dict[str, Any]:
        """
        Optimize the knowledge base for better performance.
        
        Returns:
            Dictionary with optimization results
        """
        start_time = time.time()
        
        results = {
            'operations_performed': [],
            'start_time': start_time,
            'end_time': None,
            'duration': None
        }
        
        try:
            # Optimize storage
            storage_optimization = self.storage.optimize_storage()
            results['operations_performed'].extend(storage_optimization['operations_performed'])
            
            # Clear retriever cache
            self.retriever.cache.clear()
            results['operations_performed'].append("Cleared retriever cache")
            
            # Rebuild knowledge graph
            graph_results = self.storage.create_knowledge_graph()
            results['operations_performed'].append(
                f"Rebuilt knowledge graph: {graph_results['nodes']} nodes, {graph_results['relationships']} relationships"
            )
            
            results['end_time'] = time.time()
            results['duration'] = results['end_time'] - start_time
            
            self.logger.info(f"Knowledge base optimization completed in {results['duration']:.4f}s")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Knowledge base optimization failed: {str(e)}")
            results['error'] = str(e)
            results['end_time'] = time.time()
            results['duration'] = results['end_time'] - start_time
            return results
    
    def generate_embeddings_for_existing_artifacts(self, limit: int = 100) -> Dict[str, Any]:
        """
        Generate embeddings for existing artifacts that don't have them.
        
        Args:
            limit: Maximum number of artifacts to process
            
        Returns:
            Dictionary with embedding generation results
        """
        start_time = time.time()
        
        results = {
            'total_artifacts_checked': 0,
            'artifacts_without_embeddings': 0,
            'embeddings_generated': 0,
            'start_time': start_time,
            'end_time': None,
            'duration': None
        }
        
        try:
            # Find artifacts without embeddings
            artifacts = self.storage.retrieve_knowledge_artifacts(
                query={'embeddings': {'$exists': False}},
                limit=limit
            )
            
            results['total_artifacts_checked'] = len(artifacts)
            results['artifacts_without_embeddings'] = len(artifacts)
            
            # Generate embeddings
            for artifact in artifacts:
                try:
                    # Generate embedding
                    embedding = self.embedding_gen.generate_knowledge_artifact_embedding(artifact)
                    
                    # Update artifact with embedding
                    artifact['embeddings'] = embedding
                    artifact['updated_at'] = datetime.now().isoformat()
                    
                    # Store updated artifact
                    self.storage.store_knowledge_artifact(artifact, generate_embedding=False)
                    
                    results['embeddings_generated'] += 1
                    
                except Exception as e:
                    self.logger.warning(f"Failed to generate embedding for artifact {artifact.get('_id')}: {str(e)}")
            
            results['end_time'] = time.time()
            results['duration'] = results['end_time'] - start_time
            
            self.logger.info(f"Generated embeddings for {results['embeddings_generated']}/{results['artifacts_without_embeddings']} artifacts")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Embedding generation failed: {str(e)}")
            results['error'] = str(e)
            results['end_time'] = time.time()
            results['duration'] = results['end_time'] - start_time
            return results
    
    def get_system_health(self) -> Dict[str, Any]:
        """
        Get overall system health and status.
        
        Returns:
            Dictionary containing system health information
        """
        try:
            # Get component statuses
            storage_stats = self.storage.get_aggregated_statistics()
            retriever_performance = self.retriever.get_performance_metrics()
            
            # Calculate health metrics
            total_artifacts = storage_stats.get('total_artifacts', 0)
            quality_score = retriever_performance.get('overall_quality_score', 0)
            
            # Determine health status
            if total_artifacts == 0:
                health_status = 'initializing'
                health_score = 0.0
            elif quality_score >= 0.8:
                health_status = 'healthy'
                health_score = 1.0
            elif quality_score >= 0.6:
                health_status = 'good'
                health_score = 0.8
            elif quality_score >= 0.4:
                health_status = 'fair'
                health_score = 0.6
            else:
                health_status = 'needs_attention'
                health_score = 0.4
            
            return {
                'status': health_status,
                'health_score': health_score,
                'components': {
                    'storage': {
                        'status': 'operational',
                        'artifacts': total_artifacts,
                        'collections': len(storage_stats.get('collection_statistics', {}))
                    },
                    'retriever': {
                        'status': 'operational',
                        'cache_hit_rate': retriever_performance.get('cache_hit_rate', 0),
                        'total_queries': retriever_performance.get('total_queries', 0)
                    },
                    'extractor': {
                        'status': 'operational'
                    },
                    'embedding_generator': {
                        'status': 'operational',
                        'models_available': len(self.embedding_gen.models)
                    }
                },
                'performance': {
                    'average_processing_time': self.performance_metrics.get('average_processing_time', 0),
                    'total_operations': self.performance_metrics.get('total_operations', 0)
                },
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get system health: {str(e)}")
            return {
                'status': 'error',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }

# Example usage and testing
async def main():
    """Example usage of the Enhanced Knowledge Engine"""
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Initialize enhanced engine
    engine = EnhancedKnowledgeEngine()
    
    print("🚀 Enhanced Knowledge Engine Example")
    print("=" * 50)
    
    # Example 1: Process workflow with enhanced features
    print("\n1. Processing workflow with enhanced features...")
    
    workflow_data = {
        'workflow_id': 'enhanced_test_001',
        'timestamp': datetime.now().isoformat(),
        'execution_data': {
            'problem_type': 'decomposition',
            'complexity': 'high',
            'team_size': 5,
            'success': True,
            'execution_time': 3600
        },
        'solution_patterns': [
            {
                'pattern': 'hierarchical_task_analysis',
                'effectiveness': 0.95,
                'context': 'complex_decomposition'
            }
        ],
        'critique_patterns': [
            {
                'pattern': 'resource_allocation',
                'issue': 'suboptimal_distribution',
                'severity': 'medium'
            }
        ],
        'team_performance': {
            'efficiency': 0.87,
            'collaboration': 0.92,
            'adaptability': 0.85
        },
        'gauntlet_effectiveness': {
            'completion_rate': 0.90,
            'quality_score': 0.88,
            'iteration_count': 3
        }
    }
    
    processing_result = engine.process_workflow_with_enhanced_features(workflow_data)
    print(f"✅ Processed workflow: {processing_result['status']}")
    print(f"✅ Extracted {processing_result['knowledge_extracted']} knowledge artifacts")
    print(f"✅ Processing time: {processing_result['processing_time']:.4f}s")
    
    # Example 2: Enhanced search
    print("\n2. Performing enhanced search...")
    
    search_results = engine.enhanced_search(
        query="complex decomposition strategies",
        query_type="hybrid",
        limit=3
    )
    print(f"✅ Found {len(search_results)} search results")
    for i, result in enumerate(search_results, 1):
        print(f"  {i}. {result.get('content', 'No content')[:60]}...")
    
    # Example 3: Personalized recommendations
    print("\n3. Getting personalized recommendations...")
    
    context = {
        'problem_type': 'decomposition',
        'complexity': 'high',
        'team_size': 5,
        'recommendation_type': 'solution_pattern'
    }
    
    user_profile = {
        'preferred_problem_types': ['decomposition', 'optimization'],
        'expertise_level': 'intermediate',
        'preferred_sources': ['workflow_execution', 'expert_analysis']
    }
    
    recommendations = engine.get_personalized_recommendations(context, user_profile)
    print(f"✅ Got {len(recommendations)} personalized recommendations")
    
    # Example 4: Knowledge analytics
    print("\n4. Generating knowledge analytics...")
    
    analytics = engine.get_knowledge_analytics()
    print(f"✅ Knowledge base contains {analytics['storage_statistics']['total_artifacts']} artifacts")
    print(f"✅ Overall quality score: {analytics['overall_quality_score']:.2f}")
    print(f"✅ Current trend: {analytics['trend_analysis'].get('trend', 'unknown')}")
    
    # Example 5: System health
    print("\n5. Checking system health...")
    
    health = engine.get_system_health()
    print(f"✅ System status: {health['status']}")
    print(f"✅ Health score: {health['health_score']:.2f}")
    
    print("\n" + "=" * 50)
    print("🎉 Enhanced Knowledge Engine example completed successfully!")

if __name__ == "__main__":
    asyncio.run(main())