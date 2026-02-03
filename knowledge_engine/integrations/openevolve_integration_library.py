"""
OpenEvolve Integration Library Integration for Knowledge Engine

This module provides integration with the OpenEvolve Integration Library,
enabling standardized access to various AI and knowledge processing components.
"""

import asyncio
import logging
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Union, Type, Callable
from dataclasses import dataclass
import uuid


logger = logging.getLogger(__name__)


@dataclass
class OEILResult:
    """Result of an OpenEvolve Integration Library operation."""
    success: bool
    output: Any
    metadata: Dict[str, Any]
    processing_time_ms: float = 0.0
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'success': self.success,
            'output': self.output,
            'metadata': self.metadata,
            'processing_time_ms': self.processing_time_ms,
            'error': self.error
        }


class OpenEvolveIntegrationLibrary:
    """
    Integration with OpenEvolve Integration Library.
    
    Provides unified access to:
    - LeanAide formal verification
    - Evolution algorithms
    - Knowledge graph operations
    - Tool creation and execution (Maker)
    - CrewAI orchestration
    - Problem decomposition
    - Solution verification
    - Assembly operations
    - Solution generation
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the OpenEvolve Integration Library.
        
        Args:
            config: Configuration for the integration library
        """
        self.config = config or self._get_default_config()
        
        # Initialize integration adapters
        self.adapters: Dict[str, Any] = {}
        self.backend_client = None
        
        # Initialize based on configuration
        self._initialize_adapters()
        
        logger.info({
            "msg": "OpenEvolveIntegrationLibrary initialized",
            "config": self.config,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for the integration library."""
        return {
            "backend_url": "http://localhost:8000",
            "api_key": None,
            "timeout": 30,
            "max_retries": 3,
            "retry_delay": 1.0,
            "circuit_breaker": {
                "enabled": True,
                "failure_threshold": 5,
                "reset_timeout": 30000,
                "success_threshold": 2
            },
            "integrations": {
                "leanaide": {
                    "enabled": True,
                    "model_type": "standard",
                    "timeout": 60
                },
                "evolution": {
                    "enabled": True,
                    "max_generations": 100,
                    "population_size": 50
                },
                "knowledge": {
                    "enabled": True,
                    "graph_type": "neo4j"
                },
                "maker": {
                    "enabled": True,
                    "max_tools": 100
                },
                "crewai": {
                    "enabled": True,
                    "max_agents": 10
                },
                "decomposition": {
                    "enabled": True,
                    "max_subproblems": 50
                },
                "verification": {
                    "enabled": True,
                    "check_types": ["logical", "mathematical", "structural"]
                },
                "assembly": {
                    "enabled": True,
                    "max_components": 100
                },
                "solution": {
                    "enabled": True,
                    "optimization_objectives": ["efficiency", "accuracy", "robustness"]
                }
            }
        }
    
    def _initialize_adapters(self):
        """Initialize integration adapters based on configuration."""
        try:
            # Import the OpenEvolve Integration Library
            from openevolve_integrations import (
                LeanAideIntegration,
                EvolutionIntegration,
                KnowledgeIntegration,
                MakerIntegration,
                CrewAIIntegration,
                DecompositionIntegration,
                VerificationIntegration,
                AssemblyIntegration,
                SolutionIntegration,
                BackendClient
            )
            
            # Initialize backend client
            self.backend_client = BackendClient(
                base_url=self.config.get("backend_url", "http://localhost:8000"),
                api_key=self.config.get("api_key"),
                timeout=self.config.get("timeout", 30)
            )
            
            # Initialize each integration based on config
            integrations_config = self.config.get("integrations", {})
            
            if integrations_config.get("leanaide", {}).get("enabled", True):
                self.adapters["leanaide"] = LeanAideIntegration(
                    client=self.backend_client,
                    retry_config={"max_attempts": self.config.get("max_retries", 3)},
                    circuit_breaker_config=self.config.get("circuit_breaker", {})
                )
            
            if integrations_config.get("evolution", {}).get("enabled", True):
                self.adapters["evolution"] = EvolutionIntegration(
                    client=self.backend_client,
                    retry_config={"max_attempts": self.config.get("max_retries", 3)},
                    circuit_breaker_config=self.config.get("circuit_breaker", {})
                )
            
            if integrations_config.get("knowledge", {}).get("enabled", True):
                self.adapters["knowledge"] = KnowledgeIntegration(
                    client=self.backend_client,
                    retry_config={"max_attempts": self.config.get("max_retries", 3)},
                    circuit_breaker_config=self.config.get("circuit_breaker", {})
                )
            
            if integrations_config.get("maker", {}).get("enabled", True):
                self.adapters["maker"] = MakerIntegration(
                    client=self.backend_client,
                    retry_config={"max_attempts": self.config.get("max_retries", 3)},
                    circuit_breaker_config=self.config.get("circuit_breaker", {})
                )
            
            if integrations_config.get("crewai", {}).get("enabled", True):
                self.adapters["crewai"] = CrewAIIntegration(
                    client=self.backend_client,
                    retry_config={"max_attempts": self.config.get("max_retries", 3)},
                    circuit_breaker_config=self.config.get("circuit_breaker", {})
                )
            
            if integrations_config.get("decomposition", {}).get("enabled", True):
                self.adapters["decomposition"] = DecompositionIntegration(
                    client=self.backend_client,
                    retry_config={"max_attempts": self.config.get("max_retries", 3)},
                    circuit_breaker_config=self.config.get("circuit_breaker", {})
                )
            
            if integrations_config.get("verification", {}).get("enabled", True):
                self.adapters["verification"] = VerificationIntegration(
                    client=self.backend_client,
                    retry_config={"max_attempts": self.config.get("max_retries", 3)},
                    circuit_breaker_config=self.config.get("circuit_breaker", {})
                )
            
            if integrations_config.get("assembly", {}).get("enabled", True):
                self.adapters["assembly"] = AssemblyIntegration(
                    client=self.backend_client,
                    retry_config={"max_attempts": self.config.get("max_retries", 3)},
                    circuit_breaker_config=self.config.get("circuit_breaker", {})
                )
            
            if integrations_config.get("solution", {}).get("enabled", True):
                self.adapters["solution"] = SolutionIntegration(
                    client=self.backend_client,
                    retry_config={"max_attempts": self.config.get("max_retries", 3)},
                    circuit_breaker_config=self.config.get("circuit_breaker", {})
                )
            
            logger.info({
                "msg": "OpenEvolve Integration Library adapters initialized",
                "adapters_count": len(self.adapters),
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            
        except ImportError:
            logger.warning({
                "msg": "OpenEvolve Integration Library not available, using mock implementation",
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            # Initialize with mock adapters
            self._initialize_mock_adapters()
        except Exception as e:
            logger.error({
                "msg": f"Failed to initialize OpenEvolve Integration Library: {e}",
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            raise
    
    def _initialize_mock_adapters(self):
        """Initialize mock adapters when the library is not available."""
        logger.warning({
            "msg": "OpenEvolve Integration Library not available - adapters will fail on use",
            "install": "pip install openevolve-lib",
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
        
        # Create failing mock implementations
        from ..optional_imports import create_failing_mock
        
        MockAdapter = create_failing_mock(
            package_name='openevolve-lib',
            feature_name='OpenEvolve Integration Adapters',
            install_command='pip install openevolve-lib'
        )
        
        self._mock_adapter_class = MockAdapter
        # Don't initialize adapters - they'll fail when accessed
        self.adapters = {}
    
    async def execute_integration(
        self,
        integration_name: str,
        operation: str,
        input_data: Any,
        config: Optional[Dict[str, Any]] = None,
        correlation_id: Optional[str] = None
    ) -> OEILResult:
        """
        Execute an operation on a specific integration.
        
        Args:
            integration_name: Name of the integration to use
            operation: Operation to perform
            input_data: Input data for the operation
            config: Configuration for the operation
            correlation_id: Correlation ID for tracking
            
        Returns:
            OEILResult with execution results
        """
        correlation_id = correlation_id or f"oeil_{integration_name}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')}"
        
        start_time = datetime.now(timezone.utc)
        
        logger.info({
            "msg": "Starting OpenEvolve Integration Library operation",
            "integration_name": integration_name,
            "operation": operation,
            "input_size": len(str(input_data)) if input_data else 0,
            "correlation_id": correlation_id,
            "timestamp": start_time.isoformat()
        })
        
        try:
            if integration_name not in self.adapters:
                raise ValueError(f"Integration '{integration_name}' not available")
            
            adapter = self.adapters[integration_name]
            
            # Prepare inputs for the adapter
            inputs = {
                "operation": operation,
                "input": input_data,
                "config": config or {}
            }
            
            # Execute the operation
            result = await adapter.execute(inputs, config)
            
            processing_time_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            
            oeil_result = OEILResult(
                success=True,
                output=result,
                metadata={
                    "integration_name": integration_name,
                    "operation": operation,
                    "processing_time_ms": processing_time_ms
                },
                processing_time_ms=processing_time_ms
            )
            
            logger.info({
                "msg": "OpenEvolve Integration Library operation completed",
                "correlation_id": correlation_id,
                "integration_name": integration_name,
                "operation": operation,
                "output_size": len(str(result)) if result else 0,
                "processing_time_ms": processing_time_ms,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            
            return oeil_result
            
        except Exception as e:
            processing_time_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            
            logger.error({
                "msg": "OpenEvolve Integration Library operation failed",
                "correlation_id": correlation_id,
                "integration_name": integration_name,
                "operation": operation,
                "error": str(e),
                "processing_time_ms": processing_time_ms,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            
            return OEILResult(
                success=False,
                output=None,
                metadata={
                    "integration_name": integration_name,
                    "operation": operation,
                    "processing_time_ms": processing_time_ms
                },
                processing_time_ms=processing_time_ms,
                error=str(e)
            )
    
    async def run_formal_verification(
        self,
        theorem: str,
        proof: Optional[str] = None,
        correlation_id: Optional[str] = None
    ) -> OEILResult:
        """
        Run formal verification using LeanAide integration.
        
        Args:
            theorem: Theorem to verify
            proof: Optional proof to verify (if not provided, attempt to generate)
            correlation_id: Correlation ID for tracking
            
        Returns:
            OEILResult with verification results
        """
        correlation_id = correlation_id or f"oeil_verify_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')}"
        
        start_time = datetime.now(timezone.utc)
        
        logger.info({
            "msg": "Starting formal verification with OpenEvolve Integration Library",
            "theorem_length": len(theorem),
            "proof_provided": proof is not None,
            "correlation_id": correlation_id,
            "timestamp": start_time.isoformat()
        })
        
        try:
            if "leanaide" not in self.adapters:
                raise RuntimeError("LeanAide integration not available")
            
            lean_adapter = self.adapters["leanaide"]
            
            # Prepare verification input
            if proof:
                # Verify existing proof
                inputs = {
                    "operation": "verify",
                    "input": proof,
                    "config": {}
                }
            else:
                # Generate and verify proof
                inputs = {
                    "operation": "prove",
                    "input": theorem,
                    "config": {"strategy": "auto", "timeout": 60}
                }
            
            result = await lean_adapter.execute(inputs)
            
            processing_time_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            
            oeil_result = OEILResult(
                success=True,
                output=result,
                metadata={
                    "integration_used": "leanaide",
                    "operation": "formal_verification",
                    "processing_time_ms": processing_time_ms
                },
                processing_time_ms=processing_time_ms
            )
            
            logger.info({
                "msg": "Formal verification with OpenEvolve Integration Library completed",
                "correlation_id": correlation_id,
                "success": result.get("success", False) if isinstance(result, dict) else True,
                "processing_time_ms": processing_time_ms,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            
            return oeil_result
            
        except Exception as e:
            processing_time_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            
            logger.error({
                "msg": "Formal verification with OpenEvolve Integration Library failed",
                "correlation_id": correlation_id,
                "error": str(e),
                "processing_time_ms": processing_time_ms,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            
            return OEILResult(
                success=False,
                output=None,
                metadata={
                    "integration_used": "leanaide",
                    "operation": "formal_verification",
                    "processing_time_ms": processing_time_ms
                },
                processing_time_ms=processing_time_ms,
                error=str(e)
            )
    
    async def extract_knowledge(
        self,
        text: str,
        extraction_type: str = "entities_relations",
        correlation_id: Optional[str] = None
    ) -> OEILResult:
        """
        Extract knowledge from text using the knowledge integration.
        
        Args:
            text: Text to extract knowledge from
            extraction_type: Type of extraction ('entities_relations', 'triples', 'graph')
            correlation_id: Correlation ID for tracking
            
        Returns:
            OEILResult with extraction results
        """
        correlation_id = correlation_id or f"oeil_kg_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')}"
        
        start_time = datetime.now(timezone.utc)
        
        logger.info({
            "msg": "Starting knowledge extraction with OpenEvolve Integration Library",
            "text_length": len(text),
            "extraction_type": extraction_type,
            "correlation_id": correlation_id,
            "timestamp": start_time.isoformat()
        })
        
        try:
            if "knowledge" not in self.adapters:
                raise RuntimeError("Knowledge integration not available")
            
            knowledge_adapter = self.adapters["knowledge"]
            
            # Prepare extraction input based on type
            if extraction_type == "entities_relations":
                operation = "extract"
                input_data = {"text": text, "extract_types": ["entities", "relations"]}
            elif extraction_type == "triples":
                operation = "extract"
                input_data = {"text": text, "extract_types": ["triples"]}
            elif extraction_type == "graph":
                operation = "extract"
                input_data = {"text": text, "extract_types": ["graph"]}
            else:
                operation = "extract"
                input_data = {"text": text}
            
            inputs = {
                "operation": operation,
                "input": input_data,
                "config": {}
            }
            
            result = await knowledge_adapter.execute(inputs)
            
            processing_time_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            
            oeil_result = OEILResult(
                success=True,
                output=result,
                metadata={
                    "integration_used": "knowledge",
                    "operation": "knowledge_extraction",
                    "extraction_type": extraction_type,
                    "processing_time_ms": processing_time_ms
                },
                processing_time_ms=processing_time_ms
            )
            
            logger.info({
                "msg": "Knowledge extraction with OpenEvolve Integration Library completed",
                "correlation_id": correlation_id,
                "extraction_type": extraction_type,
                "output_size": len(str(result)) if result else 0,
                "processing_time_ms": processing_time_ms,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            
            return oeil_result
            
        except Exception as e:
            processing_time_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            
            logger.error({
                "msg": "Knowledge extraction with OpenEvolve Integration Library failed",
                "correlation_id": correlation_id,
                "error": str(e),
                "processing_time_ms": processing_time_ms,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            
            return OEILResult(
                success=False,
                output=None,
                metadata={
                    "integration_used": "knowledge",
                    "operation": "knowledge_extraction",
                    "extraction_type": extraction_type,
                    "processing_time_ms": processing_time_ms
                },
                processing_time_ms=processing_time_ms,
                error=str(e)
            )
    
    async def create_tool(
        self,
        tool_spec: Dict[str, Any],
        correlation_id: Optional[str] = None
    ) -> OEILResult:
        """
        Create a tool using the maker integration.
        
        Args:
            tool_spec: Specification for the tool to create
            correlation_id: Correlation ID for tracking
            
        Returns:
            OEILResult with tool creation results
        """
        correlation_id = correlation_id or f"oeil_tool_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')}"
        
        start_time = datetime.now(timezone.utc)
        
        logger.info({
            "msg": "Starting tool creation with OpenEvolve Integration Library",
            "tool_spec_size": len(str(tool_spec)),
            "correlation_id": correlation_id,
            "timestamp": start_time.isoformat()
        })
        
        try:
            if "maker" not in self.adapters:
                raise RuntimeError("Maker integration not available")
            
            maker_adapter = self.adapters["maker"]
            
            inputs = {
                "operation": "create",
                "input": tool_spec,
                "config": {}
            }
            
            result = await maker_adapter.execute(inputs)
            
            processing_time_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            
            oeil_result = OEILResult(
                success=True,
                output=result,
                metadata={
                    "integration_used": "maker",
                    "operation": "tool_creation",
                    "processing_time_ms": processing_time_ms
                },
                processing_time_ms=processing_time_ms
            )
            
            logger.info({
                "msg": "Tool creation with OpenEvolve Integration Library completed",
                "correlation_id": correlation_id,
                "success": result.get("success", False) if isinstance(result, dict) else True,
                "processing_time_ms": processing_time_ms,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            
            return oeil_result
            
        except Exception as e:
            processing_time_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            
            logger.error({
                "msg": "Tool creation with OpenEvolve Integration Library failed",
                "correlation_id": correlation_id,
                "error": str(e),
                "processing_time_ms": processing_time_ms,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            
            return OEILResult(
                success=False,
                output=None,
                metadata={
                    "integration_used": "maker",
                    "operation": "tool_creation",
                    "processing_time_ms": processing_time_ms
                },
                processing_time_ms=processing_time_ms,
                error=str(e)
            )
    
    async def delegate_task(
        self,
        task_description: str,
        agent_preferences: Optional[Dict[str, Any]] = None,
        correlation_id: Optional[str] = None
    ) -> OEILResult:
        """
        Delegate a task using the CrewAI integration.
        
        Args:
            task_description: Description of the task to delegate
            agent_preferences: Preferences for agent selection
            correlation_id: Correlation ID for tracking
            
        Returns:
            OEILResult with delegation results
        """
        correlation_id = correlation_id or f"oeil_delegate_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')}"
        
        start_time = datetime.now(timezone.utc)
        
        logger.info({
            "msg": "Starting task delegation with OpenEvolve Integration Library",
            "task_description_length": len(task_description),
            "correlation_id": correlation_id,
            "timestamp": start_time.isoformat()
        })
        
        try:
            if "crewai" not in self.adapters:
                raise RuntimeError("CrewAI integration not available")
            
            crewai_adapter = self.adapters["crewai"]
            
            inputs = {
                "operation": "delegate",
                "input": {
                    "task": task_description,
                    "preferences": agent_preferences or {}
                },
                "config": {}
            }
            
            result = await crewai_adapter.execute(inputs)
            
            processing_time_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            
            oeil_result = OEILResult(
                success=True,
                output=result,
                metadata={
                    "integration_used": "crewai",
                    "operation": "task_delegation",
                    "processing_time_ms": processing_time_ms
                },
                processing_time_ms=processing_time_ms
            )
            
            logger.info({
                "msg": "Task delegation with OpenEvolve Integration Library completed",
                "correlation_id": correlation_id,
                "success": result.get("success", False) if isinstance(result, dict) else True,
                "processing_time_ms": processing_time_ms,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            
            return oeil_result
            
        except Exception as e:
            processing_time_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            
            logger.error({
                "msg": "Task delegation with OpenEvolve Integration Library failed",
                "correlation_id": correlation_id,
                "error": str(e),
                "processing_time_ms": processing_time_ms,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            
            return OEILResult(
                success=False,
                output=None,
                metadata={
                    "integration_used": "crewai",
                    "operation": "task_delegation",
                    "processing_time_ms": processing_time_ms
                },
                processing_time_ms=processing_time_ms,
                error=str(e)
            )
    
    async def decompose_problem(
        self,
        problem_statement: str,
        strategy: str = "functional",
        correlation_id: Optional[str] = None
    ) -> OEILResult:
        """
        Decompose a problem using the decomposition integration.
        
        Args:
            problem_statement: Statement of the problem to decompose
            strategy: Strategy for decomposition
            correlation_id: Correlation ID for tracking
            
        Returns:
            OEILResult with decomposition results
        """
        correlation_id = correlation_id or f"oeil_decomp_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')}"
        
        start_time = datetime.now(timezone.utc)
        
        logger.info({
            "msg": "Starting problem decomposition with OpenEvolve Integration Library",
            "problem_statement_length": len(problem_statement),
            "strategy": strategy,
            "correlation_id": correlation_id,
            "timestamp": start_time.isoformat()
        })
        
        try:
            if "decomposition" not in self.adapters:
                raise RuntimeError("Decomposition integration not available")
            
            decomp_adapter = self.adapters["decomposition"]
            
            inputs = {
                "operation": "decompose",
                "input": {
                    "problem": problem_statement,
                    "strategy": strategy
                },
                "config": {}
            }
            
            result = await decomp_adapter.execute(inputs)
            
            processing_time_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            
            oeil_result = OEILResult(
                success=True,
                output=result,
                metadata={
                    "integration_used": "decomposition",
                    "operation": "problem_decomposition",
                    "strategy": strategy,
                    "processing_time_ms": processing_time_ms
                },
                processing_time_ms=processing_time_ms
            )
            
            logger.info({
                "msg": "Problem decomposition with OpenEvolve Integration Library completed",
                "correlation_id": correlation_id,
                "success": result.get("success", False) if isinstance(result, dict) else True,
                "processing_time_ms": processing_time_ms,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            
            return oeil_result
            
        except Exception as e:
            processing_time_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            
            logger.error({
                "msg": "Problem decomposition with OpenEvolve Integration Library failed",
                "correlation_id": correlation_id,
                "error": str(e),
                "processing_time_ms": processing_time_ms,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            
            return OEILResult(
                success=False,
                output=None,
                metadata={
                    "integration_used": "decomposition",
                    "operation": "problem_decomposition",
                    "strategy": strategy,
                    "processing_time_ms": processing_time_ms
                },
                processing_time_ms=processing_time_ms,
                error=str(e)
            )
    
    async def verify_solution(
        self,
        solution: Any,
        requirements: List[str],
        correlation_id: Optional[str] = None
    ) -> OEILResult:
        """
        Verify a solution using the verification integration.
        
        Args:
            solution: Solution to verify
            requirements: Requirements to check against
            correlation_id: Correlation ID for tracking
            
        Returns:
            OEILResult with verification results
        """
        correlation_id = correlation_id or f"oeil_verify_sol_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')}"
        
        start_time = datetime.now(timezone.utc)
        
        logger.info({
            "msg": "Starting solution verification with OpenEvolve Integration Library",
            "solution_size": len(str(solution)) if solution else 0,
            "requirements_count": len(requirements),
            "correlation_id": correlation_id,
            "timestamp": start_time.isoformat()
        })
        
        try:
            if "verification" not in self.adapters:
                raise RuntimeError("Verification integration not available")
            
            verification_adapter = self.adapters["verification"]
            
            inputs = {
                "operation": "verify",
                "input": {
                    "solution": solution,
                    "requirements": requirements
                },
                "config": {}
            }
            
            result = await verification_adapter.execute(inputs)
            
            processing_time_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            
            oeil_result = OEILResult(
                success=True,
                output=result,
                metadata={
                    "integration_used": "verification",
                    "operation": "solution_verification",
                    "requirements_count": len(requirements),
                    "processing_time_ms": processing_time_ms
                },
                processing_time_ms=processing_time_ms
            )
            
            logger.info({
                "msg": "Solution verification with OpenEvolve Integration Library completed",
                "correlation_id": correlation_id,
                "success": result.get("success", False) if isinstance(result, dict) else True,
                "processing_time_ms": processing_time_ms,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            
            return oeil_result
            
        except Exception as e:
            processing_time_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            
            logger.error({
                "msg": "Solution verification with OpenEvolve Integration Library failed",
                "correlation_id": correlation_id,
                "error": str(e),
                "processing_time_ms": processing_time_ms,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            
            return OEILResult(
                success=False,
                output=None,
                metadata={
                    "integration_used": "verification",
                    "operation": "solution_verification",
                    "requirements_count": len(requirements),
                    "processing_time_ms": processing_time_ms
                },
                processing_time_ms=processing_time_ms,
                error=str(e)
            )
    
    async def get_integration_status(self) -> Dict[str, Any]:
        """
        Get the status of all available integrations.
        
        Returns:
            Dictionary with status information for each integration
        """
        status = {
            "available_integrations": list(self.adapters.keys()),
            "integration_details": {},
            "overall_status": "operational" if self.adapters else "degraded",
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        # Check health of each integration
        for name, adapter in self.adapters.items():
            try:
                if hasattr(adapter, 'health_check'):
                    health_result = await adapter.health_check()
                    status["integration_details"][name] = health_result
                else:
                    # For mock adapters, return basic status
                    status["integration_details"][name] = {
                        "name": name,
                        "status": "available",
                        "response_time": 0.0
                    }
            except Exception as e:
                status["integration_details"][name] = {
                    "name": name,
                    "status": "error",
                    "error": str(e)
                }
                status["overall_status"] = "degraded"
        
        return status
    
    async def close(self):
        """Close resources used by the integration."""
        logger.info({
            "msg": "Closing OpenEvolve Integration Library resources",
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
        
        # Close backend client if it has a close method
        if self.backend_client and hasattr(self.backend_client, 'close'):
            try:
                await self.backend_client.close()
            except Exception as e:
                logger.error(f"Error closing backend client: {e}")
        
        logger.info({
            "msg": "OpenEvolve Integration Library resources closed",
            "timestamp": datetime.now(timezone.utc).isoformat()
        })