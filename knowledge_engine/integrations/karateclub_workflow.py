"""
KarateClub Workflow Integration

Integrates KarateClub analytics into Knowledge Engine workflows for:
- Workflow execution analysis
- Team performance analysis
- Knowledge graph analysis

Follows CLAUDE.md principles:
- Runtime Truth: Validates workflow data at runtime
- Configuration Explicitness: All parameters via config
- Law of Idempotency: Safe to run multiple times
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
from dataclasses import dataclass, field

import networkx as nx
import numpy as np

from knowledge_engine.integrations.karateclub_analytics import (
    KarateClubAnalytics,
    CommunityResult,
    GraphMetrics,
    StructureAnalysis
)

logger = logging.getLogger(__name__)


@dataclass
class WorkflowAnalysis:
    """Result from workflow execution analysis"""
    workflow_id: str
    execution_graph: nx.Graph
    agent_communities: CommunityResult
    execution_patterns: Dict[str, Any]
    critical_path: List[str]
    bottlenecks: List[str]
    insights: List[str]
    execution_time_ms: float = 0.0


@dataclass
class TeamAnalysis:
    """Result from team performance analysis"""
    team_id: str
    collaboration_graph: nx.Graph
    sub_communities: CommunityResult
    key_contributors: List[Dict[str, Any]]
    communication_patterns: Dict[str, Any]
    performance_metrics: Dict[str, Any]
    recommendations: List[str]
    execution_time_ms: float = 0.0


@dataclass
class KGAnalysis:
    """Result from knowledge graph analysis"""
    graph_id: str
    knowledge_domains: CommunityResult
    key_concepts: List[Dict[str, Any]]
    topic_density: Dict[str, float]
    structural_insights: Dict[str, Any]
    execution_time_ms: float = 0.0


class KarateClubWorkflowIntegration:
    """
    Integrates KarateClub analytics into Knowledge Engine workflows.

    Features:
    - Workflow execution graph analysis
    - Team collaboration analysis
    - Knowledge graph structure analysis
    - Pattern detection
    - Performance insights
    """

    def __init__(
        self,
        knowledge_engine,
        analytics: Optional[KarateClubAnalytics] = None
    ):
        """
        Initialize KarateClub Workflow Integration.

        Args:
            knowledge_engine: Knowledge Engine instance
            analytics: Optional KarateClubAnalytics instance (created if not provided)
        """
        self.engine = knowledge_engine
        self.analytics = analytics or KarateClubAnalytics()

        logger.info("KarateClub Workflow Integration initialized")

    async def analyze_workflow_execution(
        self,
        workflow_data: Dict[str, Any],
        build_graph: bool = True
    ) -> WorkflowAnalysis:
        """
        Analyze workflow execution using KarateClub.

        Process:
        1. Build execution graph from workflow
        2. Detect agent communities
        3. Generate embeddings
        4. Compute metrics
        5. Identify patterns

        Args:
            workflow_data: Workflow execution data
            build_graph: Whether to build execution graph

        Returns:
            WorkflowAnalysis with insights
        """
        start_time = datetime.utcnow()

        workflow_id = workflow_data.get('workflow_id', 'unknown')
        logger.info(f"Analyzing workflow execution: {workflow_id}")

        try:
            # 1. Build execution graph
            if build_graph:
                execution_graph = await self._build_execution_graph(workflow_data)
            else:
                execution_graph = workflow_data.get('execution_graph', nx.DiGraph())

            # 2. Detect agent communities
            agent_communities = await self.analytics.detect_communities(
                execution_graph,
                algorithm='label_propagation'
            )

            # 3. Compute graph metrics
            metrics = await self.analytics.compute_graph_metrics(execution_graph)

            # 4. Analyze execution patterns
            execution_patterns = await self._analyze_execution_patterns(
                execution_graph,
                workflow_data
            )

            # 5. Find critical path
            critical_path = await self._find_critical_path(execution_graph)

            # 6. Identify bottlenecks
            bottlenecks = await self._identify_bottlenecks(
                execution_graph,
                metrics
            )

            # 7. Generate insights
            insights = await self._generate_workflow_insights(
                workflow_id,
                agent_communities,
                metrics,
                execution_patterns,
                critical_path,
                bottlenecks
            )

            elapsed_ms = (datetime.utcnow() - start_time).total_seconds() * 1000

            return WorkflowAnalysis(
                workflow_id=workflow_id,
                execution_graph=execution_graph,
                agent_communities=agent_communities,
                execution_patterns=execution_patterns,
                critical_path=critical_path,
                bottlenecks=bottlenecks,
                insights=insights,
                execution_time_ms=elapsed_ms
            )

        except Exception as e:
            logger.error(f"Workflow execution analysis failed: {e}")
            raise

    async def analyze_team_performance(
        self,
        team_data: Dict[str, Any],
        historical_data: Optional[List[Dict[str, Any]]] = None
    ) -> TeamAnalysis:
        """
        Analyze team performance using graph analytics.

        Analysis:
        1. Build collaboration graph
        2. Detect team sub-communities
        3. Identify key contributors (centrality)
        4. Analyze communication patterns
        5. Compare with historical data

        Args:
            team_data: Team performance data
            historical_data: Optional historical performance data

        Returns:
            TeamAnalysis with insights
        """
        start_time = datetime.utcnow()

        team_id = team_data.get('team_id', 'unknown')
        logger.info(f"Analyzing team performance: {team_id}")

        try:
            # 1. Build collaboration graph
            collaboration_graph = await self._build_collaboration_graph(team_data)

            # 2. Detect sub-communities
            sub_communities = await self.analytics.detect_communities(
                collaboration_graph,
                algorithm='gemsec'
            )

            # 3. Identify key contributors
            key_contributors = await self._identify_key_contributors(
                collaboration_graph
            )

            # 4. Analyze communication patterns
            communication_patterns = await self._analyze_communication_patterns(
                collaboration_graph
            )

            # 5. Compute performance metrics
            performance_metrics = await self._compute_team_metrics(
                collaboration_graph,
                team_data
            )

            # 6. Generate recommendations
            recommendations = await self._generate_team_recommendations(
                sub_communities,
                key_contributors,
                communication_patterns,
                performance_metrics
            )

            # 7. Compare with historical data if available
            if historical_data:
                comparison = await self._compare_with_historical(
                    performance_metrics,
                    historical_data
                )
                performance_metrics['historical_comparison'] = comparison

            elapsed_ms = (datetime.utcnow() - start_time).total_seconds() * 1000

            return TeamAnalysis(
                team_id=team_id,
                collaboration_graph=collaboration_graph,
                sub_communities=sub_communities,
                key_contributors=key_contributors,
                communication_patterns=communication_patterns,
                performance_metrics=performance_metrics,
                recommendations=recommendations,
                execution_time_ms=elapsed_ms
            )

        except Exception as e:
            logger.error(f"Team performance analysis failed: {e}")
            raise

    async def analyze_knowledge_graph(
        self,
        graph: nx.Graph,
        analysis_depth: str = 'standard'
    ) -> KGAnalysis:
        """
        Analyze knowledge graph structure.

        Analysis:
        1. Community detection (knowledge domains)
        2. Centrality analysis (key concepts)
        3. Clustering analysis (topic density)
        4. Graph embedding (similarity search)

        Args:
            graph: Knowledge graph
            analysis_depth: 'quick', 'standard', or 'deep'

        Returns:
            KGAnalysis with insights
        """
        start_time = datetime.utcnow()

        graph_id = graph.graph.get('id', 'unknown') if hasattr(graph, 'graph') else 'unknown'
        logger.info(f"Analyzing knowledge graph: {graph_id} (depth: {analysis_depth})")

        try:
            # 1. Community detection (knowledge domains)
            if analysis_depth == 'quick':
                community_algorithm = 'label_propagation'
            elif analysis_depth == 'deep':
                community_algorithm = 'gemsec'
            else:
                community_algorithm = 'node2vec'

            knowledge_domains = await self.analytics.detect_communities(
                graph,
                algorithm=community_algorithm
            )

            # 2. Centrality analysis (key concepts)
            key_concepts = await self._identify_key_concepts(
                graph,
                top_k=20
            )

            # 3. Clustering analysis (topic density)
            topic_density = await self._analyze_topic_density(
                graph,
                knowledge_domains
            )

            # 4. Structural insights
            structural_insights = await self._generate_structural_insights(
                graph,
                knowledge_domains,
                key_concepts,
                topic_density
            )

            elapsed_ms = (datetime.utcnow() - start_time).total_seconds() * 1000

            return KGAnalysis(
                graph_id=graph_id,
                knowledge_domains=knowledge_domains,
                key_concepts=key_concepts,
                topic_density=topic_density,
                structural_insights=structural_insights,
                execution_time_ms=elapsed_ms
            )

        except Exception as e:
            logger.error(f"Knowledge graph analysis failed: {e}")
            raise

    # ========== Helper Methods ==========

    async def _build_execution_graph(self, workflow_data: Dict[str, Any]) -> nx.Graph:
        """Build execution graph from workflow data"""
        graph = nx.DiGraph()

        # Add nodes (tasks/agents)
        tasks = workflow_data.get('tasks', [])
        for task in tasks:
            task_id = str(task.get('id', f"task_{len(graph.nodes)}"))
            graph.add_node(
                task_id,
                task_type=task.get('type', 'unknown'),
                agent=task.get('agent', 'unknown'),
                duration=task.get('duration', 0),
                status=task.get('status', 'unknown')
            )

        # Add edges (dependencies)
        dependencies = workflow_data.get('dependencies', [])
        for dep in dependencies:
            source = str(dep.get('source'))
            target = str(dep.get('target'))
            if source in graph.nodes and target in graph.nodes:
                graph.add_edge(source, target, type='dependency')

        return graph

    async def _analyze_execution_patterns(
        self,
        graph: nx.Graph,
        workflow_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze execution patterns"""
        patterns = {
            'sequential_execution': 0,
            'parallel_execution': 0,
            'avg_out_degree': 0,
            'avg_in_degree': 0,
            'max_path_length': 0
        }

        if graph.number_of_nodes() > 0:
            # Degree analysis
            out_degrees = [d for n, d in graph.out_degree()]
            in_degrees = [d for n, d in graph.in_degree()]

            patterns['avg_out_degree'] = np.mean(out_degrees) if out_degrees else 0
            patterns['avg_in_degree'] = np.mean(in_degrees) if in_degrees else 0

            # Path length
            try:
                if nx.is_weakly_connected(graph):
                    longest_path = max(nx.all_simple_paths(
                        graph.to_undirected(),
                        list(graph.nodes())[0],
                        list(graph.nodes())[-1]
                    ), key=len)
                    patterns['max_path_length'] = len(longest_path)
            except:
                pass

        return patterns

    async def _find_critical_path(self, graph: nx.Graph) -> List[str]:
        """Find critical path in execution graph"""
        try:
            # Use longest path as critical path
            if not graph.is_directed():
                graph = graph.to_directed()

            # Find all source nodes (no incoming edges)
            sources = [n for n in graph.nodes() if graph.in_degree(n) == 0]

            # Find all sink nodes (no outgoing edges)
            sinks = [n for n in graph.nodes() if graph.out_degree(n) == 0]

            # Find longest path
            longest_path = []
            for source in sources:
                for sink in sinks:
                    try:
                        paths = nx.all_simple_paths(graph, source, sink)
                        for path in paths:
                            if len(path) > len(longest_path):
                                longest_path = path
                    except:
                        pass

            return [str(node) for node in longest_path]

        except Exception as e:
            logger.warning(f"Failed to find critical path: {e}")
            return []

    async def _identify_bottlenecks(
        self,
        graph: nx.Graph,
        metrics: GraphMetrics
    ) -> List[str]:
        """Identify bottlenecks in execution"""
        bottlenecks = []

        try:
            # Find nodes with high betweenness centrality
            betweenness = nx.betweenness_centrality(graph)

            # Get top 5% nodes
            threshold = np.percentile(list(betweenness.values()), 95)

            for node, centrality in betweenness.items():
                if centrality >= threshold:
                    bottlenecks.append(str(node))

        except Exception as e:
            logger.warning(f"Failed to identify bottlenecks: {e}")

        return bottlenecks

    async def _generate_workflow_insights(
        self,
        workflow_id: str,
        communities: CommunityResult,
        metrics: GraphMetrics,
        patterns: Dict[str, Any],
        critical_path: List[str],
        bottlenecks: List[str]
    ) -> List[str]:
        """Generate insights from workflow analysis"""
        insights = []

        # Community insights
        if communities.num_communities > 1:
            insights.append(
                f"Workflow has {communities.num_communities} distinct agent communities "
                f"with modularity {communities.modularity:.2f}"
            )

        # Critical path insights
        if critical_path:
            insights.append(
                f"Critical path contains {len(critical_path)} tasks"
            )

        # Bottleneck insights
        if bottlenecks:
            insights.append(
                f"Identified {len(bottlenecks)} potential bottlenecks in execution"
            )

        # Connectivity insights
        if not metrics.is_connected:
            insights.append(
                f"Workflow graph has {metrics.num_components} disconnected components"
            )

        # Density insights
        if metrics.density < 0.1:
            insights.append("Low workflow density suggests sequential execution")
        elif metrics.density > 0.5:
            insights.append("High workflow density suggests highly parallel execution")

        return insights

    async def _build_collaboration_graph(self, team_data: Dict[str, Any]) -> nx.Graph:
        """Build collaboration graph from team data"""
        graph = nx.Graph()

        # Add nodes (team members)
        members = team_data.get('members', [])
        for member in members:
            member_id = str(member.get('id', f"member_{len(graph.nodes)}"))
            graph.add_node(
                member_id,
                name=member.get('name', 'Unknown'),
                role=member.get('role', 'Unknown'),
                contributions=member.get('contributions', 0)
            )

        # Add edges (collaborations)
        collaborations = team_data.get('collaborations', [])
        for collab in collaborations:
            member1 = str(collab.get('member1'))
            member2 = str(collab.get('member2'))
            if member1 in graph.nodes and member2 in graph.nodes:
                weight = collab.get('frequency', 1)
                if graph.has_edge(member1, member2):
                    graph[member1][member2]['weight'] += weight
                else:
                    graph.add_edge(member1, member2, weight=weight)

        return graph

    async def _identify_key_contributors(
        self,
        graph: nx.Graph
    ) -> List[Dict[str, Any]]:
        """Identify key contributors using centrality measures"""
        key_contributors = []

        try:
            # Compute centrality measures
            degree_centrality = nx.degree_centrality(graph)
            betweenness_centrality = nx.betweenness_centrality(graph)
            pagerank = nx.pagerank(graph)

            # Compute composite score
            for node in graph.nodes():
                score = (
                    0.4 * degree_centrality.get(node, 0) +
                    0.4 * betweenness_centrality.get(node, 0) +
                    0.2 * pagerank.get(node, 0)
                )

                key_contributors.append({
                    'node': str(node),
                    'score': float(score),
                    'name': graph.nodes[node].get('name', 'Unknown'),
                    'role': graph.nodes[node].get('role', 'Unknown')
                })

            # Sort by score
            key_contributors.sort(key=lambda x: x['score'], reverse=True)

        except Exception as e:
            logger.warning(f"Failed to identify key contributors: {e}")

        return key_contributors[:10]  # Top 10

    async def _analyze_communication_patterns(
        self,
        graph: nx.Graph
    ) -> Dict[str, Any]:
        """Analyze communication patterns"""
        patterns = {
            'avg_degree': 0,
            'density': 0,
            'clustering': 0,
            'num_components': 0
        }

        try:
            if graph.number_of_nodes() > 0:
                patterns['avg_degree'] = np.mean([d for n, d in graph.degree()])
                patterns['density'] = nx.density(graph)
                patterns['clustering'] = nx.average_clustering(graph)
                patterns['num_components'] = nx.number_connected_components(graph)

        except Exception as e:
            logger.warning(f"Failed to analyze communication patterns: {e}")

        return patterns

    async def _compute_team_metrics(
        self,
        graph: nx.Graph,
        team_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Compute team performance metrics"""
        metrics = {
            'team_size': graph.number_of_nodes(),
            'collaboration_count': graph.number_of_edges(),
            'avg_contributions': 0,
            'efficiency_score': 0.0
        }

        try:
            # Compute average contributions
            contributions = [
                graph.nodes[n].get('contributions', 0)
                for n in graph.nodes()
            ]
            metrics['avg_contributions'] = np.mean(contributions) if contributions else 0

            # Compute efficiency score (collaborations per team member)
            if metrics['team_size'] > 0:
                metrics['efficiency_score'] = metrics['collaboration_count'] / metrics['team_size']

        except Exception as e:
            logger.warning(f"Failed to compute team metrics: {e}")

        return metrics

    async def _generate_team_recommendations(
        self,
        sub_communities: CommunityResult,
        key_contributors: List[Dict[str, Any]],
        communication_patterns: Dict[str, Any],
        performance_metrics: Dict[str, Any]
    ) -> List[str]:
        """Generate team performance recommendations"""
        recommendations = []

        # Community recommendations
        if sub_communities.num_communities > 3:
            recommendations.append(
                "Consider cross-community team building activities to reduce silos"
            )

        # Key contributor recommendations
        if key_contributors:
            top_contributor = key_contributors[0]
            recommendations.append(
                f"Recognize and reward top contributor: {top_contributor['name']} "
                f"(score: {top_contributor['score']:.2f})"
            )

        # Communication recommendations
        if communication_patterns['density'] < 0.2:
            recommendations.append(
                "Low communication density - encourage more collaboration"
            )

        if communication_patterns['num_components'] > 1:
            recommendations.append(
                "Team has disconnected components - improve cross-team communication"
            )

        # Performance recommendations
        if performance_metrics['efficiency_score'] < 1.0:
            recommendations.append(
                "Low collaboration efficiency - review team structure"
            )

        return recommendations

    async def _compare_with_historical(
        self,
        current_metrics: Dict[str, Any],
        historical_data: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Compare current performance with historical data"""
        comparison = {
            'trend': 'stable',
            'improvement_areas': [],
            'strengths': []
        }

        # Simple comparison logic
        if len(historical_data) > 0:
            historical_avg = np.mean([
                h.get('efficiency_score', 0) for h in historical_data
            ])

            current = current_metrics.get('efficiency_score', 0)

            if current > historical_avg * 1.1:
                comparison['trend'] = 'improving'
            elif current < historical_avg * 0.9:
                comparison['trend'] = 'declining'

        return comparison

    async def _identify_key_concepts(
        self,
        graph: nx.Graph,
        top_k: int = 20
    ) -> List[Dict[str, Any]]:
        """Identify key concepts using centrality"""
        key_concepts = []

        try:
            # Compute centrality measures
            degree_centrality = nx.degree_centrality(graph)
            betweenness_centrality = nx.betweenness_centrality(graph)
            pagerank = nx.pagerank(graph)

            # Compute composite score
            for node in graph.nodes():
                score = (
                    0.3 * degree_centrality.get(node, 0) +
                    0.4 * betweenness_centrality.get(node, 0) +
                    0.3 * pagerank.get(node, 0)
                )

                key_concepts.append({
                    'concept': str(node),
                    'score': float(score),
                    'degree': degree_centrality.get(node, 0),
                    'betweenness': betweenness_centrality.get(node, 0),
                    'pagerank': pagerank.get(node, 0)
                })

            # Sort by score
            key_concepts.sort(key=lambda x: x['score'], reverse=True)

        except Exception as e:
            logger.warning(f"Failed to identify key concepts: {e}")

        return key_concepts[:top_k]

    async def _analyze_topic_density(
        self,
        graph: nx.Graph,
        knowledge_domains: CommunityResult
    ) -> Dict[str, float]:
        """Analyze topic density by community"""
        topic_density = {}

        try:
            for comm_id, members in knowledge_domains.communities.items():
                # Compute density within community
                if len(members) > 1:
                    subgraph = graph.subgraph(members)
                    density = nx.density(subgraph)
                    topic_density[comm_id] = float(density)
                else:
                    topic_density[comm_id] = 0.0

        except Exception as e:
            logger.warning(f"Failed to analyze topic density: {e}")

        return topic_density

    async def _generate_structural_insights(
        self,
        graph: nx.Graph,
        knowledge_domains: CommunityResult,
        key_concepts: List[Dict[str, Any]],
        topic_density: Dict[str, float]
    ) -> Dict[str, Any]:
        """Generate structural insights about knowledge graph"""
        insights = {
            'graph_type': 'unknown',
            'density_level': 'medium',
            'community_structure': 'weak',
            'key_findings': []
        }

        try:
            # Determine graph type
            if nx.is_directed(graph):
                insights['graph_type'] = 'directed'
            else:
                insights['graph_type'] = 'undirected'

            # Density level
            density = nx.density(graph)
            if density < 0.1:
                insights['density_level'] = 'sparse'
            elif density > 0.5:
                insights['density_level'] = 'dense'

            # Community structure
            if knowledge_domains.num_communities > 1:
                if knowledge_domains.modularity > 0.7:
                    insights['community_structure'] = 'strong'
                elif knowledge_domains.modularity > 0.4:
                    insights['community_structure'] = 'moderate'

            # Key findings
            insights['key_findings'].append(
                f"Knowledge graph contains {graph.number_of_nodes()} concepts "
                f"and {graph.number_of_edges()} relationships"
            )

            if knowledge_domains.num_communities > 1:
                insights['key_findings'].append(
                    f"Identified {knowledge_domains.num_communities} knowledge domains "
                    f"(modularity: {knowledge_domains.modularity:.2f})"
                )

            if key_concepts:
                insights['key_findings'].append(
                    f"Top concept: {key_concepts[0]['concept']} "
                    f"(centrality score: {key_concepts[0]['score']:.2f})"
                )

        except Exception as e:
            logger.warning(f"Failed to generate structural insights: {e}")

        return insights
