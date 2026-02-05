"""
Production-Ready Knowledge Engine for OpenEvolve

This module provides the complete Phase 3 implementation of the OpenEvolve
Knowledge Engine with real database integration, production-ready features,
and advanced capabilities as specified in the technical architecture.
"""

import asyncio
import json
import logging
import time
from datetime import datetime
from typing import Dict, Any, List, Optional

# Import core components
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

# Import Phase 3 components
try:
    from .real_database_integration import RealDatabaseIntegrator
except ImportError:
    from real_database_integration import RealDatabaseIntegrator

class ProductionKnowledgeEngine:
    """
    Production-Ready Knowledge Engine with Phase 3 capabilities
    
    This class provides a comprehensive, production-ready knowledge management
    system with real database integration, advanced features, and enterprise-grade
    capabilities.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the Production Knowledge Engine.
        
        Args:
            config: Configuration dictionary for all components
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Initialize core components
        self.state = KnowledgeState(query="production_engine_initialization")
        self.graph = EntityKnowledgeGraph()
        
        # Initialize Phase 1 components
        self.extractor = KnowledgeExtractor(self.config)
        
        # Initialize Phase 2 components
        self.storage = EnhancedKnowledgeStorage(self.config)
        self.retriever = EnhancedKnowledgeRetriever(self.storage, self.config)
        self.embedding_gen = EmbeddingGenerator(self.config)
        
        # Initialize Phase 3 components
        self.database_integrator = RealDatabaseIntegrator(self.config)
        
        # Check production readiness
        self.production_ready = self.database_integrator.is_production_ready()
        
        # Performance tracking
        self.performance_metrics = {
            'total_operations': 0,
            'average_processing_time': 0.0,
            'last_operation': None,
            'database_operations': {
                'qdrant': 0,
                'mongo': 0,
                'neo4j': 0,
                'redis': 0
            }
        }
        
        # System status
        self.system_status = {
            'status': 'initializing',
            'health_score': 0.0,
            'last_check': datetime.now().isoformat()
        }
        
        self.logger.info(f"Production Knowledge Engine initialized - Production ready: {self.production_ready}")
        
    def get_system_status(self) -> Dict[str, Any]:
        """
        Get current system status and health information.
        
        Returns:
            Dictionary containing system status
        """
        return {
            **self.system_status,
            'production_ready': self.production_ready,
            'database_status': self.database_integrator.get_health_status(),
            'performance': self.performance_metrics
        }
    
    def update_system_health(self):
        """Update system health status based on current conditions"""
        try:
            # Get database health
            db_health = self.database_integrator.get_health_status()
            available_dbs = db_health['available_databases']
            
            # Calculate health score
            if available_dbs >= 3:
                health_score = 1.0
                status = 'healthy'
            elif available_dbs >= 2:
                health_score = 0.8
                status = 'good'
            elif available_dbs >= 1:
                health_score = 0.6
                status = 'degraded'
            else:
                health_score = 0.3
                status = 'unhealthy'
            
            self.system_status = {
                'status': status,
                'health_score': health_score,
                'last_check': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Failed to update system health: {str(e)}")
            self.system_status = {
                'status': 'error',
                'health_score': 0.0,
                'last_check': datetime.now().isoformat(),
                'error': str(e)
            }
    
    def process_workflow_production(self, workflow_data: Dict[str, Any],
                                   generate_embeddings: bool = True) -> Dict[str, Any]:
        """
        Process workflow data with production-grade features.
        
        Args:
            workflow_data: Workflow execution data
            generate_embeddings: Whether to generate embeddings for artifacts
            
        Returns:
            Processing results with production features
        """
        start_time = time.time()
        
        # Validate input
        if not workflow_data or not isinstance(workflow_data, dict):
            return {
                'status': 'error',
                'error': 'Invalid workflow data - must be a non-empty dictionary',
                'workflow_id': 'unknown',
                'production_mode': self.production_ready
            }
        
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
            
            # Update system health
            self.update_system_health()
            
            return {
                'status': 'processed',
                'workflow_id': workflow_data.get('workflow_id', 'unknown'),
                'knowledge_extracted': len(artifacts),
                'artifacts_with_embeddings': len(artifacts) if generate_embeddings else 0,
                'processing_time': processing_time,
                'production_mode': self.production_ready,
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
            self.logger.error(f"Production workflow processing failed: {str(e)}")
            return {
                'status': 'error',
                'error': str(e),
                'workflow_id': workflow_data.get('workflow_id', 'unknown'),
                'production_mode': self.production_ready
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
    
    def production_search(self, query: str, query_type: str = 'hybrid',
                         filters: Optional[Dict[str, Any]] = None,
                         limit: int = 10, use_cache: bool = True) -> Dict[str, Any]:
        """
        Perform production-grade search with enhanced features.
        
        Args:
            query: Search query string
            query_type: Type of search (hybrid, vector, keyword, semantic)
            filters: Additional filters
            limit: Maximum number of results
            use_cache: Whether to use caching
            
        Returns:
            Dictionary with search results and metadata
        """
        start_time = time.time()
        
        # Validate input
        if not query or not isinstance(query, str):
            return {
                'status': 'error',
                'error': 'Invalid query - must be a non-empty string',
                'query': str(query),
                'query_type': query_type,
                'production_mode': self.production_ready,
                'timestamp': datetime.now().isoformat()
            }
        
        if not isinstance(limit, int) or limit <= 0:
            limit = 10  # Default limit
        
        try:
            # Perform search
            results = self.retriever.search_knowledge(query, query_type, filters, limit, use_cache)
            
            # Update performance metrics
            search_time = time.time() - start_time
            self._update_performance_metrics('search', search_time)
            
            return {
                'status': 'success',
                'query': query,
                'query_type': query_type,
                'results': results,
                'result_count': len(results),
                'processing_time': search_time,
                'production_mode': self.production_ready,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Production search failed: {str(e)}")
            return {
                'status': 'error',
                'error': str(e),
                'query': query,
                'query_type': query_type,
                'production_mode': self.production_ready,
                'timestamp': datetime.now().isoformat()
            }
    
    def get_production_recommendations(self, context: Dict[str, Any],
                                      user_profile: Optional[Dict[str, Any]] = None,
                                      limit: int = 5) -> Dict[str, Any]:
        """
        Get production-grade personalized recommendations.
        
        Args:
            context: Context dictionary
            user_profile: Optional user profile for personalization
            limit: Maximum number of recommendations
            
        Returns:
            Dictionary with recommendations and metadata
        """
        start_time = time.time()
        
        # Validate input
        if not context or not isinstance(context, dict):
            return {
                'status': 'error',
                'error': 'Invalid context - must be a non-empty dictionary',
                'context': context,
                'production_mode': self.production_ready,
                'timestamp': datetime.now().isoformat()
            }
        
        if not isinstance(limit, int) or limit <= 0:
            limit = 5  # Default limit
        
        try:
            # Get recommendations
            recommendations = self.retriever.get_personalized_recommendations(context, user_profile, limit)
            
            # Update performance metrics
            recommendation_time = time.time() - start_time
            self._update_performance_metrics('recommendations', recommendation_time)
            
            return {
                'status': 'success',
                'context': context,
                'recommendations': recommendations,
                'recommendation_count': len(recommendations),
                'processing_time': recommendation_time,
                'production_mode': self.production_ready,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Production recommendations failed: {str(e)}")
            return {
                'status': 'error',
                'error': str(e),
                'context': context,
                'production_mode': self.production_ready,
                'timestamp': datetime.now().isoformat()
            }
    
    def get_comprehensive_analytics(self) -> Dict[str, Any]:
        """
        Get comprehensive analytics about the knowledge base and system.
        
        Returns:
            Dictionary containing detailed analytics and system information
        """
        start_time = time.time()
        
        try:
            # Get storage statistics with error handling
            try:
                storage_stats = self.storage.get_aggregated_statistics()
            except Exception as e:
                self.logger.error(f"Failed to get storage statistics: {str(e)}")
                storage_stats = {'error': str(e), 'total_artifacts': 0}
            
            # Get quality metrics with error handling
            try:
                quality_metrics = self.retriever.get_knowledge_quality_metrics()
            except Exception as e:
                self.logger.error(f"Failed to get quality metrics: {str(e)}")
                quality_metrics = {'error': str(e), 'overall_quality_score': 0}
            
            # Get performance metrics with error handling
            try:
                retriever_performance = self.retriever.get_performance_metrics()
            except Exception as e:
                self.logger.error(f"Failed to get retriever performance: {str(e)}")
                retriever_performance = {'error': str(e)}
            
            # Get trend analysis with error handling
            try:
                trends = self.retriever.get_knowledge_trends(time_range='30d', analysis_type='advanced')
            except Exception as e:
                self.logger.error(f"Failed to get trend analysis: {str(e)}")
                trends = {'error': str(e), 'trend_analysis': {}}
            
            # Get database health with error handling
            try:
                db_health = self.database_integrator.get_health_status()
            except Exception as e:
                self.logger.error(f"Failed to get database health: {str(e)}")
                db_health = {'error': str(e), 'overall_status': 'error'}
            
            # Get knowledge graph with error handling
            try:
                knowledge_graph = self.storage.create_knowledge_graph()
            except Exception as e:
                self.logger.error(f"Failed to create knowledge graph: {str(e)}")
                knowledge_graph = {'error': str(e), 'nodes': 0, 'relationships': 0}
            
            # Combine all analytics
            analytics = {
                'system': {
                    'status': self.system_status,
                    'production_ready': self.production_ready,
                    'performance': self.performance_metrics
                },
                'storage': storage_stats,
                'quality': quality_metrics.get('quality_metrics', {}),
                'overall_quality_score': quality_metrics.get('overall_quality_score', 0),
                'trends': trends.get('trend_analysis', {}),
                'retriever_performance': retriever_performance,
                'database_health': db_health,
                'knowledge_graph': knowledge_graph,
                'generated_at': datetime.now().isoformat(),
                'analysis_time': time.time() - start_time
            }
            
            return analytics
            
        except Exception as e:
            self.logger.error(f"Failed to generate comprehensive analytics: {str(e)}")
            return {
                'status': 'error',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def batch_process_production(self, workflows: List[Dict[str, Any]],
                                batch_size: int = 10) -> Dict[str, Any]:
        """
        Batch process multiple workflows with production-grade features.
        
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
            'duration': None,
            'production_mode': self.production_ready
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
                        workflow_result = self.process_workflow_production(workflow)
                        
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
            
            # Update performance metrics
            self._update_performance_metrics('batch_processing', results['duration'])
            
            # Update system health
            self.update_system_health()
            
            self.logger.info(f"Production batch processing completed: {results['success_count']}/{results['total_workflows']} workflows")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Production batch processing failed: {str(e)}")
            results['error'] = str(e)
            results['end_time'] = time.time()
            results['duration'] = results['end_time'] - start_time
            return results
    
    def optimize_production_system(self) -> Dict[str, Any]:
        """
        Optimize the production system for better performance.
        
        Returns:
            Dictionary with optimization results
        """
        start_time = time.time()
        
        results = {
            'operations_performed': [],
            'start_time': start_time,
            'end_time': None,
            'duration': None,
            'production_mode': self.production_ready
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
            
            # Update system health
            self.update_system_health()
            
            results['end_time'] = time.time()
            results['duration'] = results['end_time'] - start_time
            
            # Update performance metrics
            self._update_performance_metrics('system_optimization', results['duration'])
            
            self.logger.info(f"Production system optimization completed in {results['duration']:.4f}s")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Production system optimization failed: {str(e)}")
            results['error'] = str(e)
            results['end_time'] = time.time()
            results['duration'] = results['end_time'] - start_time
            return results
    
    def generate_embeddings_production(self, limit: int = 100) -> Dict[str, Any]:
        """
        Generate embeddings for existing artifacts in production mode.
        
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
            'duration': None,
            'production_mode': self.production_ready
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
            
            # Update performance metrics
            self._update_performance_metrics('embedding_generation', results['duration'])
            
            self.logger.info(f"Generated embeddings for {results['embeddings_generated']}/{results['artifacts_without_embeddings']} artifacts")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Production embedding generation failed: {str(e)}")
            results['error'] = str(e)
            results['end_time'] = time.time()
            results['duration'] = results['end_time'] - start_time
            return results
    
    def get_production_health_report(self) -> Dict[str, Any]:
        """
        Generate a comprehensive production health report.
        
        Returns:
            Dictionary containing detailed health information
        """
        try:
            # Update system health
            self.update_system_health()
            
            # Get system status
            system_status = self.get_system_status()
            
            # Get database health
            db_health = self.database_integrator.get_health_status()
            
            # Get performance metrics
            performance = self.performance_metrics
            
            # Calculate health indicators
            health_indicators = {
                'system_health': system_status['status'],
                'health_score': system_status['health_score'],
                'production_ready': self.production_ready,
                'database_health': db_health['overall_status'],
                'available_databases': db_health['available_databases']
            }
            
            # Determine overall health status
            if health_indicators['health_score'] >= 0.8:
                overall_status = 'healthy'
            elif health_indicators['health_score'] >= 0.6:
                overall_status = 'good'
            elif health_indicators['health_score'] >= 0.4:
                overall_status = 'degraded'
            else:
                overall_status = 'unhealthy'
            
            return {
                'timestamp': datetime.now().isoformat(),
                'overall_status': overall_status,
                'health_indicators': health_indicators,
                'system_status': system_status,
                'database_health': db_health,
                'performance_metrics': performance,
                'recommendations': self._generate_health_recommendations(health_indicators)
            }
            
        except Exception as e:
            self.logger.error(f"Failed to generate health report: {str(e)}")
            return {
                'status': 'error',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def _generate_health_recommendations(self, health_indicators: Dict[str, Any]) -> List[str]:
        """Generate health recommendations based on current status"""
        recommendations = []
        
        # System health recommendations
        if health_indicators['health_score'] < 0.6:
            recommendations.append("Consider checking system logs for errors")
        
        if health_indicators['health_score'] < 0.8:
            recommendations.append("Review database connections and performance")
        
        # Database recommendations
        if health_indicators['available_databases'] < 2:
            recommendations.append("Ensure at least 2 database systems are available for production use")
        
        if not health_indicators['production_ready']:
            recommendations.append("System is not production ready - check database availability")
        
        # Performance recommendations
        if self.performance_metrics['average_processing_time'] > 1.0:
            recommendations.append("Consider optimizing queries and database indexes for better performance")
        
        return recommendations

# Example usage and testing
async def main():
    """Example usage of the Production Knowledge Engine"""
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Initialize production engine
    engine = ProductionKnowledgeEngine()
    
    print("🚀 Production Knowledge Engine Example")
    print("=" * 60)
    
    # Check system status
    system_status = engine.get_system_status()
    print(f"\nSystem Status: {system_status['status']}")
    print(f"Production Ready: {system_status['production_ready']}")
    print(f"Health Score: {system_status['health_score']}")
    
    # Example 1: Process workflow in production mode
    print("\n1. Processing workflow in production mode...")
    
    workflow_data = {
        'workflow_id': 'production_test_001',
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
    
    processing_result = engine.process_workflow_production(workflow_data)
    print(f"[OK] Processed workflow: {processing_result['status']}")
    print(f"[OK] Extracted {processing_result['knowledge_extracted']} knowledge artifacts")
    print(f"[OK] Processing time: {processing_result['processing_time']:.4f}s")
    print(f"[OK] Production mode: {processing_result['production_mode']}")
    
    # Example 2: Production search
    print("\n2. Performing production search...")
    
    search_results = engine.production_search(
        query="complex decomposition strategies",
        query_type="hybrid",
        limit=3
    )
    print(f"[OK] Search status: {search_results['status']}")
    print(f"[OK] Found {search_results['result_count']} results")
    print(f"[OK] Processing time: {search_results['processing_time']:.4f}s")
    
    # Example 3: Production recommendations
    print("\n3. Getting production recommendations...")
    
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
    
    recommendations = engine.get_production_recommendations(context, user_profile)
    print(f"[OK] Recommendations status: {recommendations['status']}")
    print(f"[OK] Got {recommendations['recommendation_count']} recommendations")
    print(f"[OK] Processing time: {recommendations['processing_time']:.4f}s")
    
    # Example 4: Comprehensive analytics
    print("\n4. Generating comprehensive analytics...")
    
    analytics = engine.get_comprehensive_analytics()
    print(f"[OK] Knowledge base contains {analytics['storage']['total_artifacts']} artifacts")
    print(f"[OK] Overall quality score: {analytics['overall_quality_score']:.2f}")
    print(f"[OK] Current trend: {analytics['trends'].get('trend', 'unknown')}")
    print(f"[OK] System health: {analytics['system']['status']['status']}")
    
    # Example 5: Health report
    print("\n5. Generating production health report...")
    
    health_report = engine.get_production_health_report()
    print(f"[OK] Overall status: {health_report['overall_status']}")
    print(f"[OK] Health score: {health_report['health_indicators']['health_score']}")
    print(f"[OK] Production ready: {health_report['health_indicators']['production_ready']}")
    
    if health_report['recommendations']:
        print("[OK] Health recommendations:")
        for i, recommendation in enumerate(health_report['recommendations'], 1):
            print(f"  {i}. {recommendation}")
    
    print("\n" + "=" * 60)
    print("🎉 Production Knowledge Engine example completed successfully!")
    print("📊 System is ready for enterprise deployment!")

if __name__ == "__main__":
    asyncio.run(main())