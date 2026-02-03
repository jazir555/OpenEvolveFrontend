"""
Knowledge Engine MCP Server

Model Context Protocol (MCP) server that exposes the Knowledge Orchestrator
and all integrated components via the MCP protocol.

Following CLAUDE.md principles:
- ZERO TRUST: Validate all inputs
- RUNTIME TRUTH: Check component availability
- IDEMPOTENCY: Safe to retry operations
- CONFIGURATION EXPLICITNESS: Explicit config required
- UTC TIME: All timestamps in UTC
- STRUCTURED LOGGING: JSON logs with correlation IDs
"""

import json
import logging
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime, timezone
from pathlib import Path

from .knowledge_orchestrator import (
    KnowledgeOrchestrator,
    OrchestratorConfig,
    DomainPresets,
    ComponentType,
    create_finance_orchestrator,
    create_chemistry_orchestrator,
    create_healthcare_orchestrator,
    create_research_orchestrator,
    create_minimal_orchestrator
)

logger = logging.getLogger(__name__)


class KnowledgeEngineMCPHandler:
    """
    MCP handler for the Knowledge Engine Orchestrator.
    
    Provides:
    - Domain-specific orchestrators (finance, chemistry, healthcare, research, minimal)
    - Component management (enable/disable/skip)
    - Pipeline execution with full configurability
    - Status monitoring and health checks
    """
    
    def __init__(self):
        """Initialize the MCP handler"""
        self.orchestrators: Dict[str, KnowledgeOrchestrator] = {}
        self.active_orchestrator_id: Optional[str] = None
        
        logger.info({
            "msg": "KnowledgeEngineMCPHandler initialized",
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
    
    def handle(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main entry point for MCP requests.
        
        Args:
            request: MCP request with 'method' and 'params'
            
        Returns:
            MCP response with result or error
        """
        method = request.get('method', '')
        params = request.get('params', {})
        request_id = request.get('id', 'unknown')
        
        logger.debug({
            "msg": "MCP request received",
            "method": method,
            "request_id": request_id,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
        
        # Map methods to handlers
        handlers: Dict[str, Callable] = {
            # Orchestrator creation
            'knowledge.create_finance_orchestrator': self._create_finance_orchestrator,
            'knowledge.create_chemistry_orchestrator': self._create_chemistry_orchestrator,
            'knowledge.create_healthcare_orchestrator': self._create_healthcare_orchestrator,
            'knowledge.create_research_orchestrator': self._create_research_orchestrator,
            'knowledge.create_minimal_orchestrator': self._create_minimal_orchestrator,
            'knowledge.create_custom_orchestrator': self._create_custom_orchestrator,
            
            # Processing
            'knowledge.process': self._process,
            'knowledge.process_with_config': self._process_with_config,
            
            # Component management
            'knowledge.enable_component': self._enable_component,
            'knowledge.disable_component': self._disable_component,
            'knowledge.get_component_status': self._get_component_status,
            
            # Status and monitoring
            'knowledge.get_orchestrator_status': self._get_orchestrator_status,
            'knowledge.list_orchestrators': self._list_orchestrators,
            'knowledge.switch_orchestrator': self._switch_orchestrator,
            'knowledge.delete_orchestrator': self._delete_orchestrator,
            
            # Direct component access
            'knowledge.extract_with_deepke': self._extract_with_deepke,
            'knowledge.analyze_graph_with_karateclub': self._analyze_graph_with_karateclub,
            'knowledge.mine_patterns_with_pami': self._mine_patterns_with_pami,
            'knowledge.embed_with_neuralkg': self._embed_with_neuralkg,
            'knowledge.discover_causal_structure': self._discover_causal_structure,
            'knowledge.analyze_attractor_landscape': self._analyze_attractor_landscape,
            'knowledge.query_chemical_knowledge': self._query_chemical_knowledge,
            'knowledge.model_dynamics_with_neuromancer': self._model_dynamics_with_neuromancer,
            
            # Health and diagnostics
            'knowledge.health_check': self._health_check,
            'knowledge.get_available_methods': self._get_available_methods,
        }
        
        handler = handlers.get(method)
        if handler:
            try:
                result = handler(params)
                return {
                    "jsonrpc": "2.0",
                    "result": result,
                    "id": request_id
                }
            except Exception as e:
                logger.error({
                    "msg": f"Error handling method {method}",
                    "error": str(e),
                    "request_id": request_id
                })
                return {
                    "jsonrpc": "2.0",
                    "error": {
                        "code": -32000,
                        "message": str(e)
                    },
                    "id": request_id
                }
        else:
            return {
                "jsonrpc": "2.0",
                "error": {
                    "code": -32601,
                    "message": f"Method not found: {method}"
                },
                "id": request_id
            }
    
    # === Orchestrator Creation Methods ===
    
    def _create_finance_orchestrator(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Create a finance-optimized orchestrator"""
        orchestrator_id = params.get('orchestrator_id', f"finance_{len(self.orchestrators)}")
        config_params = params.get('config', {})
        
        orchestrator = create_finance_orchestrator(**config_params)
        self.orchestrators[orchestrator_id] = orchestrator
        self.active_orchestrator_id = orchestrator_id
        
        return {
            "orchestrator_id": orchestrator_id,
            "domain": "finance",
            "status": "created",
            "components": list(orchestrator.components.keys())
        }
    
    def _create_chemistry_orchestrator(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Create a chemistry-optimized orchestrator"""
        orchestrator_id = params.get('orchestrator_id', f"chemistry_{len(self.orchestrators)}")
        config_params = params.get('config', {})
        
        orchestrator = create_chemistry_orchestrator(**config_params)
        self.orchestrators[orchestrator_id] = orchestrator
        self.active_orchestrator_id = orchestrator_id
        
        return {
            "orchestrator_id": orchestrator_id,
            "domain": "chemistry",
            "status": "created",
            "components": list(orchestrator.components.keys())
        }
    
    def _create_healthcare_orchestrator(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Create a healthcare-optimized orchestrator"""
        orchestrator_id = params.get('orchestrator_id', f"healthcare_{len(self.orchestrators)}")
        config_params = params.get('config', {})
        
        orchestrator = create_healthcare_orchestrator(**config_params)
        self.orchestrators[orchestrator_id] = orchestrator
        self.active_orchestrator_id = orchestrator_id
        
        return {
            "orchestrator_id": orchestrator_id,
            "domain": "healthcare",
            "status": "created",
            "components": list(orchestrator.components.keys())
        }
    
    def _create_research_orchestrator(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Create a research-optimized orchestrator (comprehensive)"""
        orchestrator_id = params.get('orchestrator_id', f"research_{len(self.orchestrators)}")
        config_params = params.get('config', {})
        
        orchestrator = create_research_orchestrator(**config_params)
        self.orchestrators[orchestrator_id] = orchestrator
        self.active_orchestrator_id = orchestrator_id
        
        return {
            "orchestrator_id": orchestrator_id,
            "domain": "research",
            "status": "created",
            "components": list(orchestrator.components.keys())
        }
    
    def _create_minimal_orchestrator(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Create a minimal orchestrator (essential components only)"""
        orchestrator_id = params.get('orchestrator_id', f"minimal_{len(self.orchestrators)}")
        config_params = params.get('config', {})
        
        orchestrator = create_minimal_orchestrator(**config_params)
        self.orchestrators[orchestrator_id] = orchestrator
        self.active_orchestrator_id = orchestrator_id
        
        return {
            "orchestrator_id": orchestrator_id,
            "domain": "general",
            "status": "created",
            "components": list(orchestrator.components.keys())
        }
    
    def _create_custom_orchestrator(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Create a custom orchestrator from configuration"""
        orchestrator_id = params.get('orchestrator_id', f"custom_{len(self.orchestrators)}")
        config_dict = params.get('config', {})
        
        config = OrchestratorConfig.from_dict(config_dict)
        orchestrator = KnowledgeOrchestrator(config)
        
        self.orchestrators[orchestrator_id] = orchestrator
        self.active_orchestrator_id = orchestrator_id
        
        return {
            "orchestrator_id": orchestrator_id,
            "domain": config.domain.value,
            "status": "created",
            "components": list(orchestrator.components.keys())
        }
    
    # === Processing Methods ===
    
    def _process(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Process data through the active orchestrator"""
        orchestrator = self._get_orchestrator(params.get('orchestrator_id'))
        input_data = params.get('data', {})
        
        result = orchestrator.process(input_data)
        return result
    
    def _process_with_config(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Process data with runtime configuration overrides"""
        orchestrator = self._get_orchestrator(params.get('orchestrator_id'))
        input_data = params.get('data', {})
        runtime_config = params.get('runtime_config', {})
        
        result = orchestrator.process(input_data, custom_config=runtime_config)
        return result
    
    # === Component Management ===
    
    def _enable_component(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Enable a component in an orchestrator"""
        orchestrator = self._get_orchestrator(params.get('orchestrator_id'))
        component_name = params.get('component')
        required = params.get('required', False)
        
        try:
            component = ComponentType(component_name)
            orchestrator.config.enable_component(component, required)
            
            # Re-initialize if component is available
            if component in orchestrator.config.components:
                orchestrator._initialize_components()
            
            return {
                "component": component_name,
                "enabled": True,
                "required": required,
                "available": component in orchestrator.components
            }
        except ValueError:
            raise ValueError(f"Unknown component: {component_name}")
    
    def _disable_component(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Disable a component in an orchestrator"""
        orchestrator = self._get_orchestrator(params.get('orchestrator_id'))
        component_name = params.get('component')
        
        try:
            component = ComponentType(component_name)
            orchestrator.config.disable_component(component)
            
            # Remove from active components
            if component in orchestrator.components:
                del orchestrator.components[component]
            
            return {
                "component": component_name,
                "enabled": False,
                "disabled_in_config": True
            }
        except ValueError:
            raise ValueError(f"Unknown component: {component_name}")
    
    def _get_component_status(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Get status of all components"""
        orchestrator = self._get_orchestrator(params.get('orchestrator_id'))
        
        component_status = {}
        for comp_type, comp_config in orchestrator.config.components.items():
            component_status[comp_type.value] = {
                "enabled": comp_config.enabled,
                "required": comp_config.required,
                "available": comp_type in orchestrator.components,
                "configured": comp_config.to_dict()
            }
        
        return {
            "components": component_status,
            "total_configured": len(orchestrator.config.components),
            "total_available": len(orchestrator.components),
            "active_components": [c.value for c in orchestrator.components.keys()]
        }
    
    # === Status and Monitoring ===
    
    def _get_orchestrator_status(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Get status of an orchestrator"""
        orchestrator = self._get_orchestrator(params.get('orchestrator_id'))
        return orchestrator.get_status()
    
    def _list_orchestrators(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """List all orchestrators"""
        orchestrator_list = []
        for orch_id, orch in self.orchestrators.items():
            status = orch.get_status()
            orchestrator_list.append({
                "id": orch_id,
                "name": status['name'],
                "domain": status['domain'],
                "initialized_components": len(status['initialized_components']),
                "is_active": orch_id == self.active_orchestrator_id
            })
        
        return {
            "orchestrators": orchestrator_list,
            "active_orchestrator": self.active_orchestrator_id,
            "total_count": len(orchestrator_list)
        }
    
    def _switch_orchestrator(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Switch the active orchestrator"""
        orchestrator_id = params.get('orchestrator_id')
        
        if orchestrator_id not in self.orchestrators:
            raise ValueError(f"Orchestrator not found: {orchestrator_id}")
        
        self.active_orchestrator_id = orchestrator_id
        
        return {
            "active_orchestrator": orchestrator_id,
            "previous_active": self.active_orchestrator_id,
            "available_orchestrators": list(self.orchestrators.keys())
        }
    
    def _delete_orchestrator(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Delete an orchestrator"""
        orchestrator_id = params.get('orchestrator_id')
        
        if orchestrator_id not in self.orchestrators:
            raise ValueError(f"Orchestrator not found: {orchestrator_id}")
        
        del self.orchestrators[orchestrator_id]
        
        # Update active orchestrator if needed
        if self.active_orchestrator_id == orchestrator_id:
            self.active_orchestrator_id = next(iter(self.orchestrators.keys()), None)
        
        return {
            "deleted": orchestrator_id,
            "remaining_orchestrators": list(self.orchestrators.keys()),
            "active_orchestrator": self.active_orchestrator_id
        }
    
    # === Direct Component Access ===
    
    def _extract_with_deepke(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Extract knowledge using DeepKE"""
        orchestrator = self._get_orchestrator(params.get('orchestrator_id'))
        text = params.get('text', '')
        
        from ..integrations import DeepKEEnhancedExtractor
        extractor = DeepKEEnhancedExtractor()
        return extractor.extract_with_deepke(text)
    
    def _analyze_graph_with_karateclub(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze graph using Karate Club"""
        orchestrator = self._get_orchestrator(params.get('orchestrator_id'))
        graph_data = params.get('graph', {})
        
        from ..integrations import KarateClubGraphAnalyzer
        analyzer = KarateClubGraphAnalyzer()
        return analyzer.analyze_graph(graph_data)
    
    def _mine_patterns_with_pami(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Mine patterns using PAMI"""
        orchestrator = self._get_orchestrator(params.get('orchestrator_id'))
        transactions = params.get('transactions', [])
        min_support = params.get('min_support', 0.1)
        
        from ..integrations import PAMIPatternMiner
        miner = PAMIPatternMiner()
        if miner is None:
            return {"status": "error", "message": "PAMI not available"}
        
        return miner.mine_frequent_patterns(transactions, min_support)
    
    def _embed_with_neuralkg(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Generate embeddings using NeuralKG"""
        orchestrator = self._get_orchestrator(params.get('orchestrator_id'))
        triples = params.get('triples', [])
        model = params.get('model', 'transe')
        
        from ..integrations import NeuralKGEmbedder
        embedder = NeuralKGEmbedder()
        if embedder is None:
            return {"status": "error", "message": "NeuralKG not available"}
        
        return embedder.generate_embeddings(triples, model_name=model)
    
    def _discover_causal_structure(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Discover causal structure using Causal-Learn"""
        orchestrator = self._get_orchestrator(params.get('orchestrator_id'))
        data = params.get('data', [])
        algorithm = params.get('algorithm', 'pc')
        
        from ..integrations import CausalDiscoveryEngine
        engine = CausalDiscoveryEngine()
        if engine is None:
            return {"status": "error", "message": "Causal-Learn not available"}
        
        import numpy as np
        return engine.discover_causal_structure(np.array(data), algorithm=algorithm)
    
    def _analyze_attractor_landscape(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze attractor landscape using Lagrange-Mapper"""
        orchestrator = self._get_orchestrator(params.get('orchestrator_id'))
        embeddings = params.get('embeddings', [])
        labels = params.get('labels', [])
        
        from ..integrations import LagrangeAttractorAnalyzer
        analyzer = LagrangeAttractorAnalyzer()
        if analyzer is None:
            return {"status": "error", "message": "Lagrange-Mapper not available"}
        
        import numpy as np
        return analyzer.analyze_embedding_landscape(np.array(embeddings), labels)
    
    def _query_chemical_knowledge(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Query chemical knowledge using GlobalChem"""
        orchestrator = self._get_orchestrator(params.get('orchestrator_id'))
        query = params.get('query', '')
        query_type = params.get('query_type', 'general')
        
        from ..integrations import GlobalChemKnowledgeAdapter
        adapter = GlobalChemKnowledgeAdapter()
        if adapter is None:
            return {"status": "error", "message": "GlobalChem not available"}
        
        if query_type == 'smiles':
            return adapter.query_by_smiles(query)
        elif query_type == 'name':
            return adapter.query_by_name(query)
        else:
            return adapter.get_compound_info(query)
    
    def _model_dynamics_with_neuromancer(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Model dynamics using Neuromancer"""
        orchestrator = self._get_orchestrator(params.get('orchestrator_id'))
        time_series = params.get('time_series', [])
        time_points = params.get('time_points', [])
        
        from ..integrations import NeuromancerDynamicsModeler
        modeler = NeuromancerDynamicsModeler()
        if modeler is None:
            return {"status": "error", "message": "Neuromancer not available"}
        
        import numpy as np
        return modeler.train_neural_ode(np.array(time_series), np.array(time_points))
    
    # === Health and Diagnostics ===
    
    def _health_check(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Perform health check on all orchestrators"""
        health_status = {
            "overall_status": "healthy",
            "orchestrators": {},
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        for orch_id, orch in self.orchestrators.items():
            status = orch.get_status()
            health_status["orchestrators"][orch_id] = {
                "status": "healthy" if status['initialized_components'] else "degraded",
                "components_available": len(status['initialized_components']),
                "components_configured": len(status['configured_components'])
            }
            
            if not status['initialized_components']:
                health_status["overall_status"] = "degraded"
        
        return health_status
    
    def _get_available_methods(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Get list of all available MCP methods"""
        methods = [
            # Orchestrator creation
            {
                "name": "knowledge.create_finance_orchestrator",
                "description": "Create a finance-optimized orchestrator (disables chemistry components)",
                "params": ["orchestrator_id", "config"]
            },
            {
                "name": "knowledge.create_chemistry_orchestrator",
                "description": "Create a chemistry-optimized orchestrator",
                "params": ["orchestrator_id", "config"]
            },
            {
                "name": "knowledge.create_healthcare_orchestrator",
                "description": "Create a healthcare-optimized orchestrator",
                "params": ["orchestrator_id", "config"]
            },
            {
                "name": "knowledge.create_research_orchestrator",
                "description": "Create a comprehensive research orchestrator (all components enabled)",
                "params": ["orchestrator_id", "config"]
            },
            {
                "name": "knowledge.create_minimal_orchestrator",
                "description": "Create a minimal orchestrator (essential components only)",
                "params": ["orchestrator_id", "config"]
            },
            {
                "name": "knowledge.create_custom_orchestrator",
                "description": "Create a custom orchestrator from configuration",
                "params": ["orchestrator_id", "config"]
            },
            # Processing
            {
                "name": "knowledge.process",
                "description": "Process data through the active orchestrator",
                "params": ["orchestrator_id", "data"]
            },
            {
                "name": "knowledge.process_with_config",
                "description": "Process data with runtime configuration overrides",
                "params": ["orchestrator_id", "data", "runtime_config"]
            },
            # Component management
            {
                "name": "knowledge.enable_component",
                "description": "Enable a component in an orchestrator",
                "params": ["orchestrator_id", "component", "required"]
            },
            {
                "name": "knowledge.disable_component",
                "description": "Disable a component in an orchestrator",
                "params": ["orchestrator_id", "component"]
            },
            {
                "name": "knowledge.get_component_status",
                "description": "Get status of all components",
                "params": ["orchestrator_id"]
            },
            # Status and monitoring
            {
                "name": "knowledge.get_orchestrator_status",
                "description": "Get status of an orchestrator",
                "params": ["orchestrator_id"]
            },
            {
                "name": "knowledge.list_orchestrators",
                "description": "List all orchestrators",
                "params": []
            },
            {
                "name": "knowledge.switch_orchestrator",
                "description": "Switch the active orchestrator",
                "params": ["orchestrator_id"]
            },
            {
                "name": "knowledge.delete_orchestrator",
                "description": "Delete an orchestrator",
                "params": ["orchestrator_id"]
            },
            # Direct component access
            {
                "name": "knowledge.extract_with_deepke",
                "description": "Extract knowledge using DeepKE",
                "params": ["orchestrator_id", "text"]
            },
            {
                "name": "knowledge.analyze_graph_with_karateclub",
                "description": "Analyze graph using Karate Club",
                "params": ["orchestrator_id", "graph"]
            },
            {
                "name": "knowledge.mine_patterns_with_pami",
                "description": "Mine patterns using PAMI",
                "params": ["orchestrator_id", "transactions", "min_support"]
            },
            {
                "name": "knowledge.embed_with_neuralkg",
                "description": "Generate embeddings using NeuralKG",
                "params": ["orchestrator_id", "triples", "model"]
            },
            {
                "name": "knowledge.discover_causal_structure",
                "description": "Discover causal structure using Causal-Learn",
                "params": ["orchestrator_id", "data", "algorithm"]
            },
            {
                "name": "knowledge.analyze_attractor_landscape",
                "description": "Analyze attractor landscape using Lagrange-Mapper",
                "params": ["orchestrator_id", "embeddings", "labels"]
            },
            {
                "name": "knowledge.query_chemical_knowledge",
                "description": "Query chemical knowledge using GlobalChem",
                "params": ["orchestrator_id", "query", "query_type"]
            },
            {
                "name": "knowledge.model_dynamics_with_neuromancer",
                "description": "Model dynamics using Neuromancer",
                "params": ["orchestrator_id", "time_series", "time_points"]
            },
            # Health and diagnostics
            {
                "name": "knowledge.health_check",
                "description": "Perform health check on all orchestrators",
                "params": []
            },
            {
                "name": "knowledge.get_available_methods",
                "description": "Get list of all available MCP methods",
                "params": []
            },
        ]
        
        return {
            "methods": methods,
            "total_methods": len(methods)
        }
    
    # === Helper Methods ===
    
    def _get_orchestrator(self, orchestrator_id: Optional[str]) -> KnowledgeOrchestrator:
        """Get an orchestrator by ID or use the active one"""
        if orchestrator_id:
            if orchestrator_id not in self.orchestrators:
                raise ValueError(f"Orchestrator not found: {orchestrator_id}")
            return self.orchestrators[orchestrator_id]
        
        if not self.orchestrators:
            # Create default minimal orchestrator
            orch_id = "default"
            self.orchestrators[orch_id] = create_minimal_orchestrator()
            self.active_orchestrator_id = orch_id
        
        if self.active_orchestrator_id is None:
            self.active_orchestrator_id = next(iter(self.orchestrators.keys()))
        
        return self.orchestrators[self.active_orchestrator_id]


def create_mcp_server() -> KnowledgeEngineMCPHandler:
    """Factory function to create the MCP server handler"""
    return KnowledgeEngineMCPHandler()
