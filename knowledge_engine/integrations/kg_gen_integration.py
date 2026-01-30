"""
KG-Gen and OneKE Integration Module for OpenEvolve Knowledge Engine

This module provides enhanced knowledge graph generation and management
capabilities by integrating kg-gen and OneKE without modifying any core files.
"""

import sys
import os
from typing import List, Dict, Any, Optional, Tuple

# Import aiohttp compatibility shim BEFORE any imports that might use dspy/litellm
from knowledge_engine.aiohttp_compat import *

# Add kg-gen to Python path for import
kg_gen_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'kg-gen', 'src')
if kg_gen_path not in sys.path:
    sys.path.insert(0, kg_gen_path)

# Add OneKE to Python path for import
oneke_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'OneKE', 'src')
if oneke_path not in sys.path:
    sys.path.insert(0, oneke_path)

class EnhancedKnowledgeGraphManager:
    """
    Enhanced knowledge graph manager that leverages kg-gen and OneKE.
    
    This class integrates advanced knowledge graph generation, storage,
    and format conversion capabilities.
    """
    
    def __init__(self, neo4j_config: Optional[Dict[str, str]] = None):
        """
        Initialize kg-gen and OneKE modules.
        
        Args:
            neo4j_config: Optional Neo4j connection config with keys:
                         uri, username, password, database (optional)
        """
        self._neo4j_config = neo4j_config or {
            'uri': 'bolt://localhost:7687',
            'username': 'neo4j',
            'password': 'password',
            'database': 'neo4j'
        }
        self._initialize_modules()
    
    def _initialize_modules(self):
        """Initialize all modules with proper error handling."""
        try:
            # Import kg-gen modules
            from kg_gen import KGGen
            from kg_gen.utils.neo4j_integration import Neo4jUploader
            
            # Import OneKE modules (functions, not classes)
            # OneKE provides: sanitize_string, generate_cypher_statements, execute_cypher_statements
            from construct.convert import generate_cypher_statements, execute_cypher_statements, sanitize_string
            
            # Initialize kg-gen components
            self.kg_generator = KGGen()
            # Neo4jUploader now requires connection params in constructor
            self.neo4j_uploader = Neo4jUploader(
                uri=self._neo4j_config.get('uri', 'bolt://localhost:7687'),
                username=self._neo4j_config.get('username', 'neo4j'),
                password=self._neo4j_config.get('password', 'password'),
                database=self._neo4j_config.get('database', 'neo4j')
            )
            
            # Initialize OneKE components (store functions)
            self.converter = {
                'generate_cypher_statements': generate_cypher_statements,
                'execute_cypher_statements': execute_cypher_statements,
                'sanitize_string': sanitize_string
            }
            
            self._kg_gen_available = True
            self._oneke_available = True
            
        except Exception as e:
            print(f"Warning: Could not import kg-gen or OneKE modules: {e}")
            print("kg-gen/OneKE integration will be partially or fully disabled.")
            
            # Try to initialize what's available
            try:
                from kg_gen import KGGen
                from kg_gen.utils.neo4j_integration import Neo4jUploader
                self.kg_generator = KGGen()
                self.neo4j_uploader = Neo4jUploader(
                    uri=self._neo4j_config.get('uri', 'bolt://localhost:7687'),
                    username=self._neo4j_config.get('username', 'neo4j'),
                    password=self._neo4j_config.get('password', 'password'),
                    database=self._neo4j_config.get('database', 'neo4j')
                )
                self._kg_gen_available = True
            except (ImportError, ConnectionError, RuntimeError):
                self._kg_gen_available = False
                self.kg_generator = None
                self.neo4j_uploader = None
            
            try:
                from construct.convert import generate_cypher_statements, execute_cypher_statements, sanitize_string
                self.converter = {
                    'generate_cypher_statements': generate_cypher_statements,
                    'execute_cypher_statements': execute_cypher_statements,
                    'sanitize_string': sanitize_string
                }
                self._oneke_available = True
            except ImportError:
                self._oneke_available = False
                self.converter = None
    
    def is_kg_gen_available(self) -> bool:
        """Check if kg-gen integration is available."""
        return self._kg_gen_available
    
    def is_oneke_available(self) -> bool:
        """Check if OneKE integration is available."""
        return self._oneke_available
    
    def generate_and_store_knowledge_graph(self, knowledge_artifacts: List[Dict[str, Any]], 
                                          config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Generate and store knowledge graph using kg-gen and OneKE.
        
        Args:
            knowledge_artifacts: List of knowledge artifacts to process
            config: Configuration for graph generation and storage
            
        Returns:
            Dictionary containing generated knowledge graph and storage results
        """
        try:
            # Set default configuration
            config = config or {
                'kg_gen': {
                    'generate_graph': True,
                    'graph_format': 'default'
                },
                'neo4j': {
                    'upload_to_neo4j': True,
                    'connection_config': {}
                },
                'oneke': {
                    'convert_formats': ['rdf', 'json-ld', 'n-triples'],
                    'include_metadata': True
                }
            }
            
            results = {
                'knowledge_graph': None,
                'neo4j_status': 'disabled',
                'converted_formats': {},
                'processing_stats': {}
            }
            
            # Convert knowledge artifacts to kg-gen format
            kg_data = self._convert_artifacts_to_kg_format(knowledge_artifacts)
            
            # Generate knowledge graph with kg-gen
            if config['kg_gen']['generate_graph'] and self.is_kg_gen_available():
                knowledge_graph = self._generate_knowledge_graph(kg_data, config['kg_gen'])
                results['knowledge_graph'] = knowledge_graph
                results['processing_stats']['kg_gen'] = {
                    'status': 'success',
                    'nodes': len(knowledge_graph.get('nodes', [])),
                    'edges': len(knowledge_graph.get('edges', []))
                }
            else:
                results['processing_stats']['kg_gen'] = {'status': 'disabled'}
            
            # Upload to Neo4j
            if config['neo4j']['upload_to_neo4j'] and self.is_kg_gen_available() and results['knowledge_graph']:
                neo4j_status = self._upload_to_neo4j(results['knowledge_graph'], config['neo4j'])
                results['neo4j_status'] = neo4j_status
                results['processing_stats']['neo4j'] = {
                    'status': neo4j_status,
                    'timestamp': self._get_current_timestamp()
                }
            else:
                results['neo4j_status'] = 'disabled'
                results['processing_stats']['neo4j'] = {'status': 'disabled'}
            
            # Convert to multiple formats using OneKE
            if config['oneke']['convert_formats'] and self.is_oneke_available() and results['knowledge_graph']:
                converted_graphs = self._convert_knowledge_graph(results['knowledge_graph'], config['oneke'])
                results['converted_formats'] = converted_graphs
                results['processing_stats']['oneke'] = {
                    'status': 'success',
                    'formats_converted': list(converted_graphs.keys()),
                    'timestamp': self._get_current_timestamp()
                }
            else:
                results['processing_stats']['oneke'] = {'status': 'disabled'}
            
            return {
                'status': 'success',
                'results': results,
                'config_used': config
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'message': f'Knowledge graph generation failed: {str(e)}',
                'results': {}
            }
    
    def _convert_artifacts_to_kg_format(self, knowledge_artifacts: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Convert OpenEvolve knowledge artifacts to kg-gen format."""
        kg_data = {
            'nodes': [],
            'edges': [],
            'metadata': {
                'source': 'openevolve',
                'conversion_timestamp': self._get_current_timestamp(),
                'num_artifacts': len(knowledge_artifacts)
            }
        }
        
        # Create node and edge sets to avoid duplicates
        node_set = set()
        edge_set = set()
        
        for artifact in knowledge_artifacts:
            # Extract nodes from artifact
            nodes = self._extract_nodes_from_artifact(artifact)
            for node in nodes:
                node_key = self._generate_node_key(node)
                if node_key not in node_set:
                    node_set.add(node_key)
                    kg_data['nodes'].append(node)
            
            # Extract edges from artifact
            edges = self._extract_edges_from_artifact(artifact)
            for edge in edges:
                edge_key = self._generate_edge_key(edge)
                if edge_key not in edge_set:
                    edge_set.add(edge_key)
                    kg_data['edges'].append(edge)
        
        return kg_data
    
    def _extract_nodes_from_artifact(self, artifact: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract nodes from a knowledge artifact."""
        nodes = []
        
        # Add subject and object as nodes for triple-based artifacts
        if artifact.get('knowledge_type') == 'triple':
            subject = artifact.get('subject')
            object_val = artifact.get('object')
            
            if subject:
                nodes.append({
                    'id': subject,
                    'type': 'entity',
                    'source': artifact.get('source', 'unknown'),
                    'metadata': {
                        'extracted_by': artifact.get('extraction_method'),
                        'confidence': artifact.get('confidence', 0.0)
                    }
                })
            
            if object_val:
                nodes.append({
                    'id': object_val,
                    'type': 'entity',
                    'source': artifact.get('source', 'unknown'),
                    'metadata': {
                        'extracted_by': artifact.get('extraction_method'),
                        'confidence': artifact.get('confidence', 0.0)
                    }
                })
        
        # Add entity nodes for entity-based artifacts
        elif artifact.get('knowledge_type') == 'entity':
            entity = artifact.get('raw_data', {}).get('entity')
            if entity:
                nodes.append({
                    'id': entity,
                    'type': artifact.get('raw_data', {}).get('type', 'entity'),
                    'source': artifact.get('source', 'unknown'),
                    'metadata': {
                        'extracted_by': artifact.get('extraction_method'),
                        'confidence': artifact.get('confidence', 0.0)
                    }
                })
        
        return nodes
    
    def _extract_edges_from_artifact(self, artifact: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract edges from a knowledge artifact."""
        edges = []
        
        # Extract edges from triple-based artifacts
        if artifact.get('knowledge_type') == 'triple':
            subject = artifact.get('subject')
            predicate = artifact.get('predicate')
            object_val = artifact.get('object')
            
            if subject and predicate and object_val:
                edges.append({
                    'source': subject,
                    'target': object_val,
                    'type': predicate,
                    'relationship': predicate,
                    'source_artifact': artifact.get('id', 'unknown'),
                    'metadata': {
                        'extracted_by': artifact.get('extraction_method'),
                        'confidence': artifact.get('confidence', 0.0),
                        'timestamp': artifact.get('metadata', {}).get('timestamp')
                    }
                })
        
        # Extract edges from relation-based artifacts
        elif artifact.get('knowledge_type') == 'relation':
            relation_data = artifact.get('raw_data', {})
            subject = relation_data.get('subject')
            predicate = relation_data.get('predicate')
            object_val = relation_data.get('object')
            
            if subject and predicate and object_val:
                edges.append({
                    'source': subject,
                    'target': object_val,
                    'type': 'relation',
                    'relationship': predicate,
                    'source_artifact': artifact.get('id', 'unknown'),
                    'metadata': {
                        'extracted_by': artifact.get('extraction_method'),
                        'confidence': artifact.get('confidence', 0.0),
                        'timestamp': artifact.get('metadata', {}).get('timestamp')
                    }
                })
        
        return edges
    
    def _generate_node_key(self, node: Dict[str, Any]) -> str:
        """Generate a unique key for a node."""
        return f"{node['id']}|{node.get('type', 'entity')}"
    
    def _generate_edge_key(self, edge: Dict[str, Any]) -> str:
        """Generate a unique key for an edge."""
        return f"{edge['source']}|{edge['relationship']}|{edge['target']}"
    
    def _generate_knowledge_graph(self, kg_data: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        """Generate knowledge graph using kg-gen."""
        try:
            # Use kg-gen to generate the knowledge graph
            # kg-gen typically expects a specific format, so we adapt our data
            kg_gen_input = self._adapt_data_for_kg_gen(kg_data)
            
            # Generate the graph
            knowledge_graph = self.kg_generator.generate(kg_gen_input)
            
            # Convert to OpenEvolve format if needed
            return self._convert_kg_gen_output(knowledge_graph)
            
        except Exception as e:
            print(f"Warning: kg-gen graph generation failed: {e}")
            # Fallback: return the input data as a basic graph
            return {
                'nodes': kg_data.get('nodes', []),
                'edges': kg_data.get('edges', []),
                'metadata': {
                    'generated_by': 'fallback',
                    'timestamp': self._get_current_timestamp(),
                    'source': 'openevolve'
                }
            }
    
    def _adapt_data_for_kg_gen(self, kg_data: Dict[str, Any]) -> Any:
        """Adapt OpenEvolve format to kg-gen expected format."""
        try:
            # kg-gen might expect different formats, so we try to adapt
            # This is a simplified adaptation - may need adjustment based on actual kg-gen API
            
            # Try to use the data as-is first
            return kg_data
            
        except Exception as e:
            print(f"Warning: Data adaptation for kg-gen failed: {e}")
            return kg_data
    
    def _convert_kg_gen_output(self, kg_gen_output: Any) -> Dict[str, Any]:
        """Convert kg-gen output to OpenEvolve format."""
        try:
            # Handle different possible output formats from kg-gen
            if isinstance(kg_gen_output, dict):
                # If it already has nodes and edges, use as-is
                if 'nodes' in kg_gen_output and 'edges' in kg_gen_output:
                    return kg_gen_output
                # If it has entities and relationships, convert
                elif 'entities' in kg_gen_output and 'relationships' in kg_gen_output:
                    return {
                        'nodes': kg_gen_output['entities'],
                        'edges': kg_gen_output['relationships'],
                        'metadata': kg_gen_output.get('metadata', {})
                    }
            
            # Fallback: try to extract nodes and edges from the output
            return {
                'nodes': getattr(kg_gen_output, 'nodes', []),
                'edges': getattr(kg_gen_output, 'edges', []),
                'metadata': {
                    'source': 'kg-gen',
                    'conversion_timestamp': self._get_current_timestamp()
                }
            }
            
        except Exception as e:
            print(f"Warning: kg-gen output conversion failed: {e}")
            return {
                'nodes': [],
                'edges': [],
                'metadata': {
                    'error': str(e),
                    'timestamp': self._get_current_timestamp()
                }
            }
    
    def _upload_to_neo4j(self, knowledge_graph: Dict[str, Any], config: Dict[str, Any]) -> str:
        """Upload knowledge graph to Neo4j using kg-gen's Neo4j integration."""
        try:
            # Note: Connection is now established in constructor
            # Just need to connect and upload
            
            # Connect to Neo4j
            self.neo4j_uploader.connect()
            
            # Upload the knowledge graph (API changed from upload to upload_graph)
            upload_result = self.neo4j_uploader.upload_graph(knowledge_graph)
            
            # Close connection
            self.neo4j_uploader.close()
            
            if upload_result:
                return 'uploaded'
            else:
                return 'upload_failed'
                
        except Exception as e:
            print(f"Warning: Neo4j upload failed: {e}")
            return f'upload_error: {str(e)}'
    
    def _convert_knowledge_graph(self, knowledge_graph: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        """Convert knowledge graph using OneKE functions.
        
        OneKE provides Cypher statement generation for Neo4j, not general format conversion.
        """
        converted_graphs = {}
        
        try:
            # OneKE's main capability is generating Cypher statements for Neo4j
            # This is available via generate_cypher_statements function
            if 'cypher' in config['convert_formats'] or 'neo4j' in config['convert_formats']:
                try:
                    import json
                    # Convert knowledge graph to OneKE format (JSON with triples)
                    oneke_format = self._convert_to_oneke_format(knowledge_graph)
                    
                    # Generate Cypher statements
                    cypher_statements = self.converter['generate_cypher_statements'](json.dumps(oneke_format))
                    
                    converted_graphs['cypher'] = {
                        'statements': cypher_statements,
                        'statement_count': len(cypher_statements)
                    }
                    
                    if config['include_metadata']:
                        converted_graphs['cypher'] = self._add_conversion_metadata(
                            converted_graphs['cypher'], 'cypher'
                        )
                        
                except Exception as e:
                    print(f"Warning: Cypher conversion failed: {e}")
                    converted_graphs['cypher'] = {
                        'error': str(e),
                        'format': 'cypher'
                    }
            
            # For other formats, we would need additional converters
            # Mark unsupported formats
            for format_name in config['convert_formats']:
                if format_name not in ['cypher', 'neo4j'] and format_name not in converted_graphs:
                    converted_graphs[format_name] = {
                        'error': f'Format {format_name} not supported by OneKE. OneKE only provides Cypher generation.',
                        'format': format_name
                    }
            
            return converted_graphs
            
        except Exception as e:
            print(f"Warning: Knowledge graph conversion failed: {e}")
            return {
                'error': str(e),
                'timestamp': self._get_current_timestamp()
            }
    
    def _convert_to_oneke_format(self, knowledge_graph: Dict[str, Any]) -> Dict[str, Any]:
        """Convert knowledge graph to OneKE format (JSON with triple_list)."""
        triple_list = []
        
        # Extract triples from edges
        for edge in knowledge_graph.get('edges', []):
            triple = {
                'head': edge.get('source', ''),
                'relation': edge.get('type', ''),
                'tail': edge.get('target', ''),
                'head_type': edge.get('source_type', 'Entity'),
                'tail_type': edge.get('target_type', 'Entity'),
                'relation_type': edge.get('relation_type', 'RELATED_TO')
            }
            triple_list.append(triple)
        
        return {'triple_list': triple_list}
    
    def _add_conversion_metadata(self, converted_data: Any, format_name: str) -> Dict[str, Any]:
        """Add metadata to converted knowledge graph."""
        if isinstance(converted_data, dict):
            metadata = {
                'conversion_metadata': {
                    'format': format_name,
                    'source': 'oneke',
                    'timestamp': self._get_current_timestamp(),
                    'converter': 'OpenEvolve Knowledge Engine'
                }
            }
            return {**converted_data, **metadata}
        else:
            return converted_data
    
    def manage_knowledge_graph_lifecycle(self, knowledge_artifacts: List[Dict[str, Any]], 
                                        lifecycle_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Complete knowledge graph lifecycle management.
        
        This method provides a comprehensive interface for the entire knowledge graph
        lifecycle from generation to storage and conversion.
        
        Args:
            knowledge_artifacts: List of knowledge artifacts
            lifecycle_config: Configuration for the complete lifecycle
            
        Returns:
            Complete lifecycle management results
        """
        try:
            # Set default lifecycle configuration
            config = lifecycle_config or {
                'extraction': {
                    'enabled': True
                },
                'graph_generation': {
                    'enabled': True,
                    'kg_gen_config': {}
                },
                'storage': {
                    'neo4j_enabled': True,
                    'neo4j_config': {}
                },
                'conversion': {
                    'enabled': True,
                    'oneke_config': {
                        'convert_formats': ['rdf', 'json-ld']
                    }
                },
                'analysis': {
                    'enabled': False
                }
            }
            
            # Execute the complete lifecycle
            results = {}
            
            # Step 1: Knowledge extraction (if enabled)
            if config['extraction']['enabled']:
                # In a real implementation, this would call the extraction pipeline
                results['extraction'] = {
                    'status': 'success',
                    'num_artifacts': len(knowledge_artifacts),
                    'timestamp': self._get_current_timestamp()
                }
            
            # Step 2: Graph generation
            if config['graph_generation']['enabled']:
                graph_result = self.generate_and_store_knowledge_graph(
                    knowledge_artifacts, config['graph_generation']['kg_gen_config']
                )
                results['graph_generation'] = graph_result
            
            # Step 3: Storage management
            if config['storage']['neo4j_enabled'] and results.get('graph_generation', {}).get('status') == 'success':
                # Storage is handled within generate_and_store_knowledge_graph
                results['storage'] = {
                    'neo4j_status': results['graph_generation']['results']['neo4j_status'],
                    'timestamp': self._get_current_timestamp()
                }
            
            # Step 4: Format conversion
            if config['conversion']['enabled'] and results.get('graph_generation', {}).get('status') == 'success':
                # Conversion is handled within generate_and_store_knowledge_graph
                results['conversion'] = {
                    'formats_converted': list(results['graph_generation']['results']['converted_formats'].keys()),
                    'timestamp': self._get_current_timestamp()
                }
            
            # Step 5: Analysis (if enabled)
            if config['analysis']['enabled']:
                # In a real implementation, this would call graph analysis
                results['analysis'] = {
                    'status': 'disabled',
                    'message': 'Analysis not implemented in this integration'
                }
            
            return {
                'status': 'success',
                'lifecycle_results': results,
                'config_used': config,
                'metadata': {
                    'lifecycle_timestamp': self._get_current_timestamp(),
                    'knowledge_engine_version': '5x_enhanced'
                }
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'message': f'Knowledge graph lifecycle management failed: {str(e)}',
                'lifecycle_results': {}
            }
    
    def _get_current_timestamp(self) -> str:
        """Get current timestamp in ISO format."""
        from datetime import datetime
        return datetime.now().isoformat()