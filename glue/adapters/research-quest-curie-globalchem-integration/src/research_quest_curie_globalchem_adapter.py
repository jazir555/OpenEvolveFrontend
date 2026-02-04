"""
Research Quest - Curie-GlobalChem Integration Adapter

This module provides an adapter that allows Research Quest to leverage the 
Curie-GlobalChem integration for conducting chemistry-related research.

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
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "glue", "adapters", "curie-globalchem-integration", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "knowledge_engine", "integrations"))

from curie_globalchem_adapter import CurieGlobalChemAdapter, create_curie_interface
from research_quest_integration import ResearchQuestIntegration, ResearchQuestResult

logger = logging.getLogger(__name__)


class ResearchQuestCurieGlobalChemAdapter:
    """
    Adapter class that bridges Research Quest with Curie-GlobalChem integration
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
        
        # Initialize the Curie-GlobalChem adapter
        curie_gc_config = self.config.get('curie_globalchem_config', {})
        self.curie_gc_adapter = CurieGlobalChemAdapter(config=curie_gc_config)
        self.curie_gc_interface = create_curie_interface(self.curie_gc_adapter)
        
        # Initialize the Research Quest integration
        research_quest_config = self.config.get('research_quest_config', {})
        self.research_quest_integration = ResearchQuestIntegration(config=research_quest_config)
        
        self.logger.info("Research Quest - Curie-GlobalChem Adapter initialized successfully")
    
    async def search_chemicals_for_research(self, research_topic: str) -> Dict[str, Any]:
        """
        Search for chemicals relevant to a research topic using GlobalChem
        
        Args:
            research_topic: The research topic to search for related chemicals
            
        Returns:
            Dictionary with research findings
        """
        self.logger.info(f"Searching for chemicals related to research topic: {research_topic}")
        
        # Parse the research topic to identify potential chemical names
        import re
        potential_chemicals = re.findall(r'\b[A-Za-z]+\b', research_topic)
        
        results = {
            'research_topic': research_topic,
            'identified_chemicals': [],
            'related_chemicals': [],
            'properties_calculated': [],
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'adapter_version': '1.0.0'
        }
        
        # Look up each potential chemical in GlobalChem
        for chem_name in potential_chemicals:
            if len(chem_name) > 2:  # Skip short words that are unlikely to be chemical names
                try:
                    chem_info = self.curie_gc_interface('search', chemical_name=chem_name)
                    if chem_info:
                        results['identified_chemicals'].append(chem_info)
                        
                        # Calculate properties if SMILES is available
                        if 'smiles' in chem_info:
                            props = self.curie_gc_adapter.get_chemical_properties(chem_info['smiles'])
                            if props:
                                results['properties_calculated'].append(props)
                        
                        # Find related chemicals
                        related = self.curie_gc_adapter.get_related_chemicals(chem_name, max_results=5)
                        results['related_chemicals'].extend(related)
                except Exception as e:
                    self.logger.warning(f"Error searching for chemical '{chem_name}': {e}")
        
        self.logger.info(f"Chemical search completed for research topic: {research_topic}")
        return results
    
    async def conduct_chemistry_research(self, research_question: str) -> Dict[str, Any]:
        """
        Conduct a chemistry-focused research using Research Quest with GlobalChem knowledge
        
        Args:
            research_question: The chemistry research question to investigate
            
        Returns:
            Dictionary with research results
        """
        self.logger.info(f"Conducting chemistry research for question: {research_question}")
        
        # Initialize Research Quest graph with the research question
        init_result = await self.research_quest_integration.initialize_graph(
            task_description=research_question
        )
        
        if not init_result.success:
            self.logger.error(f"Failed to initialize Research Quest graph: {init_result.error}")
            return {
                'success': False,
                'error': init_result.error,
                'research_question': research_question,
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
        
        # Search for relevant chemicals using GlobalChem
        chem_results = await self.search_chemicals_for_research(research_question)
        
        # Decompose the research task into dimensions
        default_dimensions = [
            "Chemical Structure Analysis",
            "Molecular Properties", 
            "Reaction Pathways",
            "Safety Considerations",
            "Environmental Impact",
            "Synthesis Methods",
            "Applications and Uses"
        ]
        
        decomp_result = await self.research_quest_integration.decompose_task(
            custom_dimensions=default_dimensions
        )
        
        if not decomp_result.success:
            self.logger.warning(f"Task decomposition failed: {decomp_result.error}")
        
        # Generate hypotheses based on chemical findings
        hypotheses = []
        for chem in chem_results['identified_chemicals']:
            hypotheses.append({
                'content': f'Hypothesis about {chem.get("name", "unknown")} chemical properties',
                'falsification_criteria': f'Test to disprove hypothesis about {chem.get("name", "unknown")}',
                'plan': {
                    'type': 'property_analysis',
                    'description': f'Analyze properties of {chem.get("name", "unknown")}',
                    'tools': ['property_calculation']
                }
            })
        
        # For each dimension, generate hypotheses if dimensions were created
        dimension_nodes = decomp_result.metadata.get('dimension_nodes', []) if decomp_result.success else []
        
        for i, dim_node in enumerate(dimension_nodes):
            if i < len(hypotheses):
                hyp_result = await self.research_quest_integration.generate_hypotheses(
                    dimension_node_id=dim_node,
                    hypotheses=[hypotheses[i]],
                    correlation_id=f"hyp_{i}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')}"
                )
                
                if not hyp_result.success:
                    self.logger.warning(f"Failed to generate hypotheses for dimension {dim_node}: {hyp_result.error}")
        
        # Extract knowledge using the full pipeline
        knowledge_result = await self.research_quest_integration.extract_knowledge(
            text=research_question,
            domain="chemistry"
        )
        
        # Compile final results
        final_results = {
            'success': True,
            'research_question': research_question,
            'chemical_findings': chem_results,
            'research_quest_results': {
                'initialization': init_result.to_dict() if hasattr(init_result, 'to_dict') else init_result,
                'decomposition': decomp_result.to_dict() if hasattr(decomp_result, 'to_dict') else decomp_result,
                'knowledge_extraction': knowledge_result.to_dict() if hasattr(knowledge_result, 'to_dict') else knowledge_result
            },
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'adapter_version': '1.0.0'
        }
        
        self.logger.info(f"Chemistry research completed for question: {research_question}")
        return final_results
    
    async def analyze_chemical_interactions(self, chemical_pairs: List[Tuple[str, str]]) -> Dict[str, Any]:
        """
        Analyze potential interactions between pairs of chemicals using GlobalChem knowledge
        
        Args:
            chemical_pairs: List of tuples containing pairs of chemical names to analyze
            
        Returns:
            Dictionary with interaction analysis results
        """
        self.logger.info(f"Analyzing interactions for {len(chemical_pairs)} chemical pairs")
        
        results = {
            'analyses': [],
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'adapter_version': '1.0.0'
        }
        
        for chem1_name, chem2_name in chemical_pairs:
            try:
                # Get information for both chemicals
                chem1_info = self.curie_gc_interface('search', chemical_name=chem1_name)
                chem2_info = self.curie_gc_interface('search', chemical_name=chem2_name)
                
                analysis = {
                    'chemical_1': chem1_info,
                    'chemical_2': chem2_info,
                    'potential_interactions': [],
                    'compatibility_notes': '',
                    'timestamp': datetime.now(timezone.utc).isoformat()
                }
                
                # If both chemicals exist in GlobalChem, we could potentially analyze interactions
                if chem1_info and chem2_info:
                    # In a real implementation, this would analyze potential reactions or interactions
                    # between the two chemicals based on their properties
                    analysis['compatibility_notes'] = f"Preliminary analysis of {chem1_name} and {chem2_name}"
                    
                    # Add to Research Quest for deeper analysis
                    interaction_research = await self.conduct_chemistry_research(
                        f"What are the potential interactions between {chem1_name} and {chem2_name}?"
                    )
                    analysis['research_analysis'] = interaction_research
                
                results['analyses'].append(analysis)
            except Exception as e:
                self.logger.warning(f"Error analyzing interaction between {chem1_name} and {chem2_name}: {e}")
                results['analyses'].append({
                    'chemical_1': {'name': chem1_name},
                    'chemical_2': {'name': chem2_name},
                    'error': str(e),
                    'timestamp': datetime.now(timezone.utc).isoformat()
                })
        
        self.logger.info(f"Completed interaction analysis for {len(chemical_pairs)} pairs")
        return results
    
    async def generate_research_proposal(self, topic: str) -> Dict[str, Any]:
        """
        Generate a research proposal combining Research Quest methodology with GlobalChem knowledge
        
        Args:
            topic: The research topic for the proposal
            
        Returns:
            Dictionary with research proposal
        """
        self.logger.info(f"Generating research proposal for topic: {topic}")
        
        # Search for relevant chemicals
        chem_findings = await self.search_chemicals_for_research(topic)
        
        # Use Research Quest to structure the research
        research_results = await self.conduct_chemistry_research(topic)
        
        proposal = {
            'topic': topic,
            'executive_summary': f'Research proposal for {topic} based on GlobalChem knowledge and Research Quest methodology',
            'chemical_findings': chem_findings,
            'research_methodology': {
                'stages_completed': research_results.get('research_quest_results', {}),
                'dimensions_identified': len(research_results.get('research_quest_results', {}).get('decomposition', {}).get('metadata', {}).get('dimension_nodes', [])),
                'hypotheses_generated': len(research_results.get('research_quest_results', {}).get('knowledge_extraction', {}).get('entities', []))
            },
            'recommended_experiments': [],
            'safety_considerations': [],
            'timeline_estimate': 'TBD',
            'resource_requirements': [],
            'expected_outcomes': [],
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'adapter_version': '1.0.0'
        }
        
        # Generate some recommended experiments based on chemical findings
        for chem in chem_findings['identified_chemicals']:
            if 'name' in chem and 'smiles' in chem:
                proposal['recommended_experiments'].append({
                    'experiment_type': 'Property Analysis',
                    'target_chemical': chem['name'],
                    'objective': f'Determine key properties of {chem["name"]} using computational methods',
                    'estimated_duration': '1-2 weeks'
                })
        
        self.logger.info(f"Research proposal generated for topic: {topic}")
        return proposal


def create_research_interface(adapter: ResearchQuestCurieGlobalChemAdapter):
    """
    Creates an interface function that Research Quest can use to access chemistry knowledge
    
    Args:
        adapter: Instance of ResearchQuestCurieGlobalChemAdapter
        
    Returns:
        Function that Research Quest can call to access chemistry knowledge
    """
    async def research_chemistry_query(query_type: str, **kwargs) -> Dict[str, Any]:
        """
        Interface function for Research Quest to query chemistry data
        
        Args:
            query_type: Type of query ('search', 'research', 'interactions', 'proposal')
            **kwargs: Query-specific parameters
            
        Returns:
            Query results
        """
        if query_type == 'search':
            research_topic = kwargs.get('research_topic')
            if not research_topic:
                raise ValueError("research_topic is required for search queries")
            return await adapter.search_chemicals_for_research(research_topic)
        
        elif query_type == 'research':
            research_question = kwargs.get('research_question')
            if not research_question:
                raise ValueError("research_question is required for research queries")
            return await adapter.conduct_chemistry_research(research_question)
        
        elif query_type == 'interactions':
            chemical_pairs = kwargs.get('chemical_pairs', [])
            return await adapter.analyze_chemical_interactions(chemical_pairs)
        
        elif query_type == 'proposal':
            topic = kwargs.get('topic')
            if not topic:
                raise ValueError("topic is required for proposal queries")
            return await adapter.generate_research_proposal(topic)
        
        else:
            raise ValueError(f"Unknown query type: {query_type}")
    
    return research_chemistry_query


# Example usage
async def main():
    # Initialize the adapter
    config = {
        'log_level': 'INFO',
        'curie_globalchem_config': {
            'log_level': 'WARNING'
        },
        'research_quest_config': {
            'model': 'openai/gpt-4o',
            'temperature': 0.1
        }
    }
    
    adapter = ResearchQuestCurieGlobalChemAdapter(config=config)
    
    # Create the interface for Research Quest
    research_interface = create_research_interface(adapter)
    
    # Example: Search for chemicals related to a research topic
    chem_results = await research_interface('search', research_topic='aspirin synthesis pathways')
    print("Chemical search results:", chem_results)
    
    # Example: Conduct a chemistry research
    research_results = await research_interface('research', research_question='What are the properties of aspirin and related compounds?')
    print("Research results:", research_results)
    
    # Example: Analyze chemical interactions
    interactions = await research_interface('interactions', chemical_pairs=[('aspirin', 'caffeine')])
    print("Interaction analysis:", interactions)
    
    # Example: Generate a research proposal
    proposal = await research_interface('proposal', topic='Development of new analgesic compounds')
    print("Research proposal:", proposal)


if __name__ == "__main__":
    asyncio.run(main())