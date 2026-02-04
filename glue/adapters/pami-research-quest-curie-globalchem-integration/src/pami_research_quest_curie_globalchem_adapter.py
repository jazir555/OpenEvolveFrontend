"""
PAMI - Research Quest - Curie-GlobalChem Integration Adapter

This module provides an adapter that allows PAMI (Pattern Analysis and Machine Intelligence)
to work with the Research Quest - Curie-GlobalChem integration for conducting advanced
scientific research with pattern analysis capabilities.

Following CLAUDE.md principles:
- ZERO TRUST: Validate all inputs and outputs
- ANTI-HALLUCINATION: Verify data integrity
- READ-ONLY STATE: Don't modify underlying systems' data
- IDEMPOTENCY: Safe to run multiple times
- CONFIGURATION EXPLICITNESS: All parameters configurable
- UTC: All timestamps in UTC
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timezone
import sys
import os

# Add paths to access the required modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "glue", "adapters", "research-quest-curie-globalchem-integration", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "knowledge_engine", "integrations"))

from research_quest_curie_globalchem_adapter import ResearchQuestCurieGlobalChemAdapter, create_research_interface
from pami_integration import PAMIIntegration

logger = logging.getLogger(__name__)


class PAMIResearchQuestCurieGlobalChemAdapter:
    """
    Adapter class that bridges PAMI with Research Quest - Curie-GlobalChem integration
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the adapter with configuration
        
        Args:
            config: Configuration dictionary with optional parameters
        """
        self.config = config or {}
        
        # Initialize logging
        log_level = self.config.get('log_level', 'INFO')
        logging.basicConfig(level=getattr(logging, log_level))
        self.logger = logging.getLogger(__name__)
        
        # Initialize the Research Quest - Curie-GlobalChem adapter
        rq_cg_config = self.config.get('research_quest_curie_globalchem_config', {})
        self.rq_cg_adapter = ResearchQuestCurieGlobalChemAdapter(config=rq_cg_config)
        self.rq_cg_interface = create_research_interface(self.rq_cg_adapter)
        
        # Initialize the PAMI integration
        pami_config = self.config.get('pami_config', {})
        self.pami_integration = PAMIIntegration(config=pami_config)
        
        self.logger.info("PAMI - Research Quest - Curie-GlobalChem Adapter initialized successfully")
    
    async def analyze_research_patterns(self, research_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze patterns in research data using PAMI
        
        Args:
            research_data: Research data to analyze for patterns
            
        Returns:
            Dictionary with pattern analysis results
        """
        self.logger.info("Analyzing patterns in research data")
        
        results = {
            'pattern_analysis': {},
            'insights': [],
            'recommendations': [],
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'adapter_version': '1.0.0'
        }
        
        try:
            # Extract transactions from research data if available
            transactions = research_data.get('transactions', [])
            
            if transactions and self.pami_integration.is_available():
                # Mine frequent patterns
                freq_patterns = self.pami_integration.mine_patterns(
                    data=transactions,
                    min_support=self.config.get('min_support', 0.1)
                )
                
                results['pattern_analysis']['frequent_patterns'] = freq_patterns
                
                # Discover association rules
                assoc_rules = self.pami_integration.discover_association_rules(
                    transactions=transactions,
                    min_support=self.config.get('min_support', 0.1),
                    min_confidence=self.config.get('min_confidence', 0.5)
                )
                
                results['pattern_analysis']['association_rules'] = assoc_rules
                
                # Generate insights from patterns
                if freq_patterns.get('status') == 'success':
                    patterns = freq_patterns.get('patterns', [])
                    if patterns:
                        results['insights'].append({
                            'type': 'frequent_patterns',
                            'summary': f"Found {len(patterns)} frequent patterns in research data",
                            'top_patterns': patterns[:5]  # Top 5 patterns
                        })
                
                if assoc_rules.get('status') == 'success':
                    rules = assoc_rules.get('rules', [])
                    if rules:
                        results['insights'].append({
                            'type': 'association_rules',
                            'summary': f"Found {len(rules)} association rules in research data",
                            'top_rules': rules[:5]  # Top 5 rules
                        })
            
            self.logger.info("Pattern analysis completed successfully")
        except Exception as e:
            self.logger.error(f"Error during pattern analysis: {e}")
            results['error'] = str(e)
        
        return results
    
    async def conduct_pattern_enriched_research(self, research_question: str) -> Dict[str, Any]:
        """
        Conduct research enriched with pattern analysis using all integrated systems
        
        Args:
            research_question: The research question to investigate
            
        Returns:
            Dictionary with comprehensive research results
        """
        self.logger.info(f"Conducting pattern-enriched research for question: {research_question}")
        
        results = {
            'research_question': research_question,
            'research_phase_results': {},
            'pattern_analysis_results': {},
            'combined_insights': [],
            'recommendations': [],
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'adapter_version': '1.0.0'
        }
        
        try:
            # Phase 1: Use Research Quest - Curie-GlobalChem to gather initial data
            self.logger.info("Phase 1: Gathering initial research data")
            initial_research = await self.rq_cg_interface('research', research_question=research_question)
            results['research_phase_results']['initial'] = initial_research
            
            # Extract relevant data for pattern analysis
            chem_findings = initial_research.get('chemical_findings', {})
            identified_chemicals = chem_findings.get('identified_chemicals', [])
            
            # Convert chemical data to transaction format for PAMI
            transactions = []
            for chem in identified_chemicals:
                if 'name' in chem and 'node' in chem:
                    # Create a transaction with chemical name and its category/node
                    transaction = [chem['name'], chem['node']]
                    if 'smiles' in chem:
                        transaction.append(chem['smiles'][:10])  # Shortened SMILES as identifier
                    transactions.append(transaction)
            
            # Phase 2: Analyze patterns in the gathered data
            self.logger.info("Phase 2: Analyzing patterns in research data")
            pattern_analysis = await self.analyze_research_patterns({
                'transactions': transactions,
                'research_data': initial_research
            })
            results['pattern_analysis_results'] = pattern_analysis
            
            # Phase 3: Use pattern insights to refine research
            self.logger.info("Phase 3: Refining research with pattern insights")
            
            # Generate recommendations based on patterns
            if 'pattern_analysis' in pattern_analysis:
                freq_patterns = pattern_analysis['pattern_analysis'].get('frequent_patterns', {})
                if freq_patterns.get('status') == 'success':
                    patterns = freq_patterns.get('patterns', [])
                    if patterns:
                        # Recommend further investigation of frequent patterns
                        for pattern in patterns[:3]:  # Top 3 patterns
                            results['recommendations'].append({
                                'type': 'investigation',
                                'target': pattern.get('pattern', []),
                                'reason': f"Frequent pattern detected with support {pattern.get('support_ratio', 0):.2f}",
                                'priority': 'high' if pattern.get('support_ratio', 0) > 0.5 else 'medium'
                            })
            
            # Phase 4: Generate combined insights
            results['combined_insights'].extend(pattern_analysis.get('insights', []))
            
            # Add pattern-based hypotheses
            if 'pattern_analysis' in pattern_analysis:
                assoc_rules = pattern_analysis['pattern_analysis'].get('association_rules', {})
                if assoc_rules.get('status') == 'success':
                    rules = assoc_rules.get('rules', [])
                    for rule in rules[:3]:  # Top 3 rules
                        results['combined_insights'].append({
                            'type': 'hypothesis',
                            'content': f"If {rule.get('antecedent', [])} then {rule.get('consequent', [])}",
                            'confidence': rule.get('confidence', 0),
                            'support': rule.get('support', 0)
                        })
            
            self.logger.info(f"Pattern-enriched research completed for question: {research_question}")
        except Exception as e:
            self.logger.error(f"Error during pattern-enriched research: {e}")
            results['error'] = str(e)
        
        return results
    
    async def analyze_chemical_knowledge_graph_patterns(self, knowledge_graph: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze patterns in chemical knowledge graphs using PAMI
        
        Args:
            knowledge_graph: Knowledge graph with nodes and edges to analyze
            
        Returns:
            Dictionary with graph pattern analysis results
        """
        self.logger.info("Analyzing patterns in chemical knowledge graph")
        
        results = {
            'graph_pattern_analysis': {},
            'discovered_relationships': [],
            'anomalies': [],
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'adapter_version': '1.0.0'
        }
        
        try:
            if self.pami_integration.is_available():
                # Analyze the knowledge graph for patterns
                graph_analysis = self.pami_integration.analyze_knowledge_graph_patterns(
                    graph_data=knowledge_graph,
                    min_support=self.config.get('min_support', 0.1)
                )
                
                results['graph_pattern_analysis'] = graph_analysis
                
                # Extract interesting findings
                if graph_analysis.get('status') == 'success':
                    patterns = graph_analysis.get('patterns', {})
                    
                    # Extract entity type patterns
                    entity_patterns = patterns.get('entity_types', [])
                    if entity_patterns:
                        results['discovered_relationships'].extend([
                            {
                                'type': 'entity_pattern',
                                'pattern': p['pattern'],
                                'frequency': p['count']
                            } for p in entity_patterns[:5]
                        ])
                    
                    # Extract relationship patterns
                    relation_patterns = patterns.get('relationship_types', [])
                    if relation_patterns:
                        results['discovered_relationships'].extend([
                            {
                                'type': 'relationship_pattern',
                                'pattern': p['pattern'],
                                'frequency': p['count']
                            } for p in relation_patterns[:5]
                        ])
                    
                    # Look for anomalies (low-frequency patterns that might be interesting)
                    all_patterns = entity_patterns + relation_patterns
                    if all_patterns:
                        # Anomalies could be patterns with very low frequency
                        low_freq_patterns = [p for p in all_patterns if p['count'] == min(p['count'] for p in all_patterns)]
                        results['anomalies'].extend([
                            {
                                'type': 'low_frequency_pattern',
                                'pattern': p['pattern'],
                                'frequency': p['count']
                            } for p in low_freq_patterns
                        ])
            
            self.logger.info("Chemical knowledge graph pattern analysis completed")
        except Exception as e:
            self.logger.error(f"Error during graph pattern analysis: {e}")
            results['error'] = str(e)
        
        return results
    
    async def generate_pattern_based_research_proposal(self, topic: str) -> Dict[str, Any]:
        """
        Generate a research proposal based on pattern analysis of existing knowledge
        
        Args:
            topic: The research topic for the proposal
            
        Returns:
            Dictionary with pattern-based research proposal
        """
        self.logger.info(f"Generating pattern-based research proposal for topic: {topic}")
        
        results = {
            'topic': topic,
            'executive_summary': f'Research proposal for {topic} based on pattern analysis of existing knowledge',
            'pattern_analysis': {},
            'research_recommendations': [],
            'methodology': {
                'approach': 'Pattern-driven research methodology',
                'tools': ['PAMI', 'Research Quest', 'Curie', 'GlobalChem'],
                'phases': []
            },
            'expected_outcomes': [],
            'risk_analysis': [],
            'timeline_estimate': 'TBD',
            'resource_requirements': [],
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'adapter_version': '1.0.0'
        }
        
        try:
            # First, conduct initial research to gather knowledge
            initial_research = await self.rq_cg_interface('research', research_question=topic)
            
            # Extract knowledge graph from research results
            # This is a simplified approach - in reality, we'd extract a proper knowledge graph
            knowledge_graph = {
                'nodes': [],
                'edges': []
            }
            
            # Add chemical entities as nodes
            chem_findings = initial_research.get('chemical_findings', {})
            identified_chemicals = chem_findings.get('identified_chemicals', [])
            
            for i, chem in enumerate(identified_chemicals):
                knowledge_graph['nodes'].append({
                    'id': f"chem_{i}",
                    'name': chem.get('name', f'chemical_{i}'),
                    'type': 'chemical',
                    'smiles': chem.get('smiles', ''),
                    'node': chem.get('node', 'unknown')
                })
            
            # Add relationships between chemicals if possible
            related_chemicals = chem_findings.get('related_chemicals', [])
            for i, rel_chem in enumerate(related_chemicals):
                knowledge_graph['nodes'].append({
                    'id': f"rel_chem_{i}",
                    'name': rel_chem.get('name', f'related_chemical_{i}'),
                    'type': 'chemical',
                    'relationship': rel_chem.get('relationship', 'unknown')
                })
                
                # Add edge connecting to original chemical if possible
                # This is a simplified approach
                if i < len(identified_chemicals):
                    knowledge_graph['edges'].append({
                        'source': f"chem_{i}",
                        'target': f"rel_chem_{i}",
                        'type': rel_chem.get('relationship', 'related_to')
                    })
            
            # Analyze patterns in the knowledge graph
            graph_pattern_analysis = await self.analyze_chemical_knowledge_graph_patterns(knowledge_graph)
            results['pattern_analysis'] = graph_pattern_analysis
            
            # Generate research recommendations based on patterns
            discovered_relationships = graph_pattern_analysis.get('discovered_relationships', [])
            for rel in discovered_relationships[:5]:  # Top 5 relationships
                results['research_recommendations'].append({
                    'focus_area': f"Investigate {rel['pattern']} pattern",
                    'rationale': f"This pattern appears {rel['frequency']} times in the knowledge graph",
                    'methodology': 'Experimental validation using Curie platform',
                    'expected_outcome': f"Better understanding of {rel['pattern']} relationships"
                })
            
            # Add expected outcomes based on pattern analysis
            if discovered_relationships:
                results['expected_outcomes'].append(
                    f"Identification of {len(discovered_relationships)} significant chemical relationship patterns"
                )
            
            # Add risk analysis based on anomalies
            anomalies = graph_pattern_analysis.get('anomalies', [])
            for anomaly in anomalies[:3]:  # Top 3 anomalies
                results['risk_analysis'].append({
                    'type': 'knowledge_gap',
                    'description': f"Low-frequency pattern '{anomaly['pattern']}' suggests potential knowledge gap",
                    'impact': 'medium',
                    'mitigation': 'Focused literature review and experimental validation'
                })
            
            # Set methodology phases
            results['methodology']['phases'] = [
                {
                    'phase': 1,
                    'title': 'Pattern Analysis',
                    'description': 'Analyze existing chemical knowledge for patterns',
                    'tools': ['PAMI']
                },
                {
                    'phase': 2,
                    'title': 'Hypothesis Generation',
                    'description': 'Generate hypotheses based on discovered patterns',
                    'tools': ['Research Quest']
                },
                {
                    'phase': 3,
                    'title': 'Experimental Design',
                    'description': 'Design experiments to validate hypotheses',
                    'tools': ['Curie', 'GlobalChem']
                },
                {
                    'phase': 4,
                    'title': 'Validation',
                    'description': 'Conduct experiments and validate findings',
                    'tools': ['Curie', 'GlobalChem']
                }
            ]
            
            self.logger.info(f"Pattern-based research proposal generated for topic: {topic}")
        except Exception as e:
            self.logger.error(f"Error generating pattern-based research proposal: {e}")
            results['error'] = str(e)
        
        return results


def create_unified_interface(adapter: PAMIResearchQuestCurieGlobalChemAdapter):
    """
    Creates an interface function that allows unified access to all integrated systems
    
    Args:
        adapter: Instance of PAMIResearchQuestCurieGlobalChemAdapter
        
    Returns:
        Function that provides unified access to all systems
    """
    async def unified_query(query_type: str, **kwargs) -> Dict[str, Any]:
        """
        Interface function for unified access to all integrated systems
        
        Args:
            query_type: Type of query ('pattern_analysis', 'enriched_research', 'graph_analysis', 'proposal')
            **kwargs: Query-specific parameters
            
        Returns:
            Query results
        """
        if query_type == 'pattern_analysis':
            research_data = kwargs.get('research_data', {})
            return await adapter.analyze_research_patterns(research_data)
        
        elif query_type == 'enriched_research':
            research_question = kwargs.get('research_question')
            if not research_question:
                raise ValueError("research_question is required for enriched research queries")
            return await adapter.conduct_pattern_enriched_research(research_question)
        
        elif query_type == 'graph_analysis':
            knowledge_graph = kwargs.get('knowledge_graph', {})
            return await adapter.analyze_chemical_knowledge_graph_patterns(knowledge_graph)
        
        elif query_type == 'proposal':
            topic = kwargs.get('topic')
            if not topic:
                raise ValueError("topic is required for proposal queries")
            return await adapter.generate_pattern_based_research_proposal(topic)
        
        else:
            raise ValueError(f"Unknown query type: {query_type}")
    
    return unified_query


# Example usage
async def main():
    # Initialize the adapter
    config = {
        'log_level': 'INFO',
        'min_support': 0.1,
        'min_confidence': 0.5,
        'research_quest_curie_globalchem_config': {
            'log_level': 'WARNING',
            'curie_globalchem_config': {
                'log_level': 'WARNING'
            },
            'research_quest_config': {
                'model': 'openai/gpt-4o',
                'temperature': 0.1
            }
        },
        'pami_config': {}
    }
    
    adapter = PAMIResearchQuestCurieGlobalChemAdapter(config=config)
    
    # Create the unified interface
    unified_interface = create_unified_interface(adapter)
    
    # Example: Analyze patterns in research data
    sample_data = {
        'transactions': [
            ['aspirin', 'pain_relief', 'acetylation'],
            ['ibuprofen', 'pain_relief', 'inflammation'],
            ['acetaminophen', 'pain_relief', 'fever_reduction'],
            ['aspirin', 'heart_health', 'blood_thinning'],
            ['statins', 'cholesterol', 'cardiovascular']
        ]
    }
    
    pattern_results = await unified_interface('pattern_analysis', research_data=sample_data)
    print("Pattern analysis results:", pattern_results)
    
    # Example: Conduct pattern-enriched research
    research_results = await unified_interface('enriched_research', research_question='What are effective pain relief medications?')
    print("Pattern-enriched research results:", research_results)
    
    # Example: Analyze chemical knowledge graph
    sample_graph = {
        'nodes': [
            {'id': 'n1', 'name': 'aspirin', 'type': 'drug'},
            {'id': 'n2', 'name': 'pain', 'type': 'symptom'},
            {'id': 'n3', 'name': 'inflammation', 'type': 'condition'}
        ],
        'edges': [
            {'source': 'n1', 'target': 'n2', 'type': 'treats'},
            {'source': 'n1', 'target': 'n3', 'type': 'reduces'}
        ]
    }
    
    graph_results = await unified_interface('graph_analysis', knowledge_graph=sample_graph)
    print("Graph analysis results:", graph_results)
    
    # Example: Generate pattern-based research proposal
    proposal = await unified_interface('proposal', topic='Novel drug combinations for pain management')
    print("Research proposal:", proposal)


if __name__ == "__main__":
    asyncio.run(main())