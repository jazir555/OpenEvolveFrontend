"""
AgentJSON Integration for OpenEvolve Knowledge Engine

This module provides integration with the AgentJSON system,
enabling robust JSON parsing and repair for agent-generated outputs.
"""

import asyncio
import logging
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass
import uuid


logger = logging.getLogger(__name__)


@dataclass
class AgentJSONResult:
    """Result of an AgentJSON operation."""
    success: bool
    parsed_data: Any
    status: str
    confidence: float
    repairs_applied: List[Dict[str, Any]]
    metadata: Dict[str, Any]
    processing_time_ms: float = 0.0
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'success': self.success,
            'parsed_data': self.parsed_data,
            'status': self.status,
            'confidence': self.confidence,
            'repairs_applied': self.repairs_applied,
            'metadata': self.metadata,
            'processing_time_ms': self.processing_time_ms,
            'error': self.error
        }


class AgentJSONIntegration:
    """
    Integration with AgentJSON for robust JSON parsing and repair.
    
    Provides methods for:
    - Parsing potentially malformed JSON from agent outputs
    - Repairing common JSON errors (unquoted keys, trailing commas, etc.)
    - Extracting JSON spans from arbitrary text
    - Providing confidence scores for repairs
    - Handling Top-K repair candidates
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the AgentJSON integration.
        
        Args:
            config: Configuration for AgentJSON components
        """
        self.config = config or self._get_default_config()
        
        # Initialize AgentJSON components
        self.repair_options = None
        self.parser = None
        
        # Initialize based on configuration
        self._initialize_components()
        
        logger.info({
            "msg": "AgentJSONIntegration initialized",
            "config": self.config,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for AgentJSON integration."""
        return {
            "mode": "auto",  # auto, strict_only, fast_repair, probabilistic
            "top_k": 5,
            "beam_width": 32,
            "max_repairs": 50,
            "deterministic_seed": 42,
            "partial_ok": True,
            "allow_llm": False,
            "llm_provider": None,
            "llm_mode": "patch_suggest",  # patch_suggest or token_suggest
            "llm_min_confidence": 0.2,
            "schema_hints": {},
            "debug": False
        }
    
    def _initialize_components(self):
        """Initialize AgentJSON components based on configuration."""
        try:
            # Import AgentJSON components
            from agentjson import RepairOptions, parse
            
            # Create repair options based on config
            self.repair_options = RepairOptions(
                mode=self.config.get("mode", "auto"),
                top_k=self.config.get("top_k", 5),
                beam_width=self.config.get("beam_width", 32),
                max_repairs=self.config.get("max_repairs", 50),
                deterministic_seed=self.config.get("deterministic_seed", 42),
                partial_ok=self.config.get("partial_ok", True),
                allow_llm=self.config.get("allow_llm", False),
                llm_mode=self.config.get("llm_mode", "patch_suggest"),
                llm_min_confidence=self.config.get("llm_min_confidence", 0.2),
                debug=self.config.get("debug", False)
            )
            
            # Store the parse function
            self.parser = parse
            
            logger.info({
                "msg": "AgentJSON components initialized successfully",
                "mode": self.config.get("mode", "auto"),
                "top_k": self.config.get("top_k", 5),
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            
        except ImportError:
            logger.warning({
                "msg": "AgentJSON not available, using mock implementation",
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            # Initialize with mock components
            self._initialize_mock_components()
        except Exception as e:
            logger.error({
                "msg": f"Failed to initialize AgentJSON components: {e}",
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            raise
    
    def _initialize_mock_components(self):
        """Initialize mock components when AgentJSON is not available."""
        logger.info({
            "msg": "Initializing mock AgentJSON components",
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
        
        # Create mock implementations
        class MockRepairOptions:
            def __init__(self, **kwargs):
                for k, v in kwargs.items():
                    setattr(self, k, v)
        
        def mock_parse(text, options=None):
            # Mock parsing implementation
            import json
            import re
            
            # Try to extract JSON from arbitrary text
            # Look for JSON-like structures
            json_pattern = r'(\{(?:[^{}]|(?R))*\}|\[(?:[^\[\]]|(?R))*\])'
            matches = re.findall(json_pattern, text, re.DOTALL)
            
            if matches:
                # Try to parse the first match
                for match in matches:
                    try:
                        # Clean up common issues
                        cleaned = match.strip()
                        # Replace single quotes with double quotes
                        cleaned = re.sub(r"'([^']*)':", r'"\1":', cleaned)
                        cleaned = re.sub(r":\s*'([^']*)'", r': "\1"', cleaned)
                        # Replace Python literals
                        cleaned = cleaned.replace('True', 'true').replace('False', 'false').replace('None', 'null')
                        
                        parsed = json.loads(cleaned)
                        return type('MockResult', (), {
                            'status': 'repaired',
                            'best': type('MockCandidate', (), {
                                'value': parsed,
                                'confidence': 0.8,
                                'cost': 1,
                                'repairs': [{'op': 'cleanup', 'span': (0, len(match)), 'note': 'Cleaned up quotes and literals'}]
                            })(),
                            'candidates': [type('MockCandidate', (), {
                                'value': parsed,
                                'confidence': 0.8,
                                'cost': 1,
                                'repairs': [{'op': 'cleanup', 'span': (0, len(match)), 'note': 'Cleaned up quotes and literals'}]
                            })()],
                            'best_index': 0,
                            'metrics': type('MockMetrics', (), {
                                'elapsed_ms': 1.0,
                                'llm_calls': 0,
                                'llm_time_ms': 0
                            })()
                        })()
                    except json.JSONDecodeError:
                        continue
            
            # If no JSON found, return a mock failure result
            return type('MockResult', (), {
                'status': 'failed',
                'best': None,
                'candidates': [],
                'best_index': -1,
                'metrics': type('MockMetrics', (), {
                    'elapsed_ms': 1.0,
                    'llm_calls': 0,
                    'llm_time_ms': 0
                })()
            })()
        
        self.repair_options = MockRepairOptions(**self.config)
        self.parser = mock_parse
    
    async def parse_json(
        self,
        text: str,
        mode: Optional[str] = None,
        top_k: Optional[int] = None,
        correlation_id: Optional[str] = None
    ) -> AgentJSONResult:
        """
        Parse JSON from text, repairing common errors.
        
        Args:
            text: Text containing JSON (potentially malformed)
            mode: Parsing mode ('auto', 'strict_only', 'fast_repair', 'probabilistic')
            top_k: Number of repair candidates to return
            correlation_id: Correlation ID for tracking
            
        Returns:
            AgentJSONResult with parsed data and repair information
        """
        correlation_id = correlation_id or f"agentjson_parse_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')}"
        
        start_time = datetime.now(timezone.utc)
        
        logger.info({
            "msg": "Starting AgentJSON parsing",
            "text_length": len(text),
            "mode": mode or self.config.get("mode", "auto"),
            "top_k": top_k or self.config.get("top_k", 5),
            "correlation_id": correlation_id,
            "timestamp": start_time.isoformat()
        })
        
        try:
            if not self.parser:
                raise RuntimeError("AgentJSON parser not initialized")
            
            # Create options for this specific call
            call_config = self.config.copy()
            if mode:
                call_config["mode"] = mode
            if top_k:
                call_config["top_k"] = top_k
            
            # Create repair options for this call
            if hasattr(self, 'RepairOptions'):
                from agentjson import RepairOptions
                options = RepairOptions(
                    mode=call_config.get("mode", "auto"),
                    top_k=call_config.get("top_k", 5),
                    beam_width=call_config.get("beam_width", 32),
                    max_repairs=call_config.get("max_repairs", 50),
                    deterministic_seed=call_config.get("deterministic_seed", 42),
                    partial_ok=call_config.get("partial_ok", True),
                    allow_llm=call_config.get("allow_llm", False),
                    llm_mode=call_config.get("llm_mode", "patch_suggest"),
                    llm_min_confidence=call_config.get("llm_min_confidence", 0.2),
                    debug=call_config.get("debug", False)
                )
            else:
                # Use existing options or mock
                options = self.repair_options
            
            # Parse the text
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.parser(text, options)
            )
            
            # Process the result
            parsed_data = None
            status = getattr(result, 'status', 'failed')
            confidence = 0.0
            repairs_applied = []
            
            if hasattr(result, 'best') and result.best:
                parsed_data = result.best.value
                confidence = getattr(result.best, 'confidence', 0.0)
                
                # Extract repair information
                if hasattr(result.best, 'repairs'):
                    for repair in result.best.repairs:
                        repairs_applied.append({
                            "operation": getattr(repair, 'op', 'unknown'),
                            "span": getattr(repair, 'span', (0, 0)),
                            "cost": getattr(repair, 'cost_delta', 0),
                            "note": getattr(repair, 'note', 'No description')
                        })
            
            processing_time_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            
            agentjson_result = AgentJSONResult(
                success=status in ['strict_ok', 'repaired', 'partial'],
                parsed_data=parsed_data,
                status=status,
                confidence=confidence,
                repairs_applied=repairs_applied,
                metadata={
                    "model_used": self.config.get("model", "unknown"),
                    "mode": mode or self.config.get("mode", "auto"),
                    "top_k": top_k or self.config.get("top_k", 5),
                    "processing_time_ms": processing_time_ms
                },
                processing_time_ms=processing_time_ms
            )
            
            logger.info({
                "msg": "AgentJSON parsing completed",
                "correlation_id": correlation_id,
                "status": status,
                "confidence": confidence,
                "repairs_count": len(repairs_applied),
                "processing_time_ms": processing_time_ms,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            
            return agentjson_result
            
        except Exception as e:
            processing_time_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            
            logger.error({
                "msg": "AgentJSON parsing failed",
                "correlation_id": correlation_id,
                "error": str(e),
                "processing_time_ms": processing_time_ms,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            
            return AgentJSONResult(
                success=False,
                parsed_data=None,
                status="failed",
                confidence=0.0,
                repairs_applied=[],
                metadata={
                    "mode": mode or self.config.get("mode", "auto"),
                    "processing_time_ms": processing_time_ms
                },
                processing_time_ms=processing_time_ms,
                error=str(e)
            )
    
    async def extract_json_span(
        self,
        text: str,
        correlation_id: Optional[str] = None
    ) -> AgentJSONResult:
        """
        Extract JSON span from arbitrary text containing markdown fences, prefixes, etc.
        
        Args:
            text: Arbitrary text that may contain JSON
            correlation_id: Correlation ID for tracking
            
        Returns:
            AgentJSONResult with extracted JSON data
        """
        correlation_id = correlation_id or f"agentjson_extract_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')}"
        
        start_time = datetime.now(timezone.utc)
        
        logger.info({
            "msg": "Starting AgentJSON span extraction",
            "text_length": len(text),
            "correlation_id": correlation_id,
            "timestamp": start_time.isoformat()
        })
        
        try:
            # Use the same parsing method but with specific options for extraction
            result = await self.parse_json(
                text=text,
                mode="auto",
                top_k=1,
                correlation_id=f"{correlation_id}_parse"
            )
            
            processing_time_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            result.metadata["processing_time_ms"] = processing_time_ms
            result.processing_time_ms = processing_time_ms
            
            logger.info({
                "msg": "AgentJSON span extraction completed",
                "correlation_id": correlation_id,
                "status": result.status,
                "success": result.success,
                "processing_time_ms": processing_time_ms,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            
            return result
            
        except Exception as e:
            processing_time_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            
            logger.error({
                "msg": "AgentJSON span extraction failed",
                "correlation_id": correlation_id,
                "error": str(e),
                "processing_time_ms": processing_time_ms,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            
            return AgentJSONResult(
                success=False,
                parsed_data=None,
                status="failed",
                confidence=0.0,
                repairs_applied=[],
                metadata={"processing_time_ms": processing_time_ms},
                processing_time_ms=processing_time_ms,
                error=str(e)
            )
    
    async def repair_json(
        self,
        json_text: str,
        repair_type: str = "auto",
        correlation_id: Optional[str] = None
    ) -> AgentJSONResult:
        """
        Repair common JSON errors in the provided text.
        
        Args:
            json_text: JSON text with potential errors
            repair_type: Type of repair ('auto', 'heuristic', 'probabilistic')
            correlation_id: Correlation ID for tracking
            
        Returns:
            AgentJSONResult with repaired JSON data
        """
        correlation_id = correlation_id or f"agentjson_repair_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')}"
        
        start_time = datetime.now(timezone.utc)
        
        logger.info({
            "msg": "Starting AgentJSON repair",
            "json_text_length": len(json_text),
            "repair_type": repair_type,
            "correlation_id": correlation_id,
            "timestamp": start_time.isoformat()
        })
        
        try:
            result = await self.parse_json(
                text=json_text,
                mode=repair_type,
                top_k=1,
                correlation_id=f"{correlation_id}_parse"
            )
            
            processing_time_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            result.metadata["processing_time_ms"] = processing_time_ms
            result.processing_time_ms = processing_time_ms
            
            logger.info({
                "msg": "AgentJSON repair completed",
                "correlation_id": correlation_id,
                "status": result.status,
                "confidence": result.confidence,
                "repairs_count": len(result.repairs_applied),
                "processing_time_ms": processing_time_ms,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            
            return result
            
        except Exception as e:
            processing_time_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            
            logger.error({
                "msg": "AgentJSON repair failed",
                "correlation_id": correlation_id,
                "error": str(e),
                "processing_time_ms": processing_time_ms,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            
            return AgentJSONResult(
                success=False,
                parsed_data=None,
                status="failed",
                confidence=0.0,
                repairs_applied=[],
                metadata={"processing_time_ms": processing_time_ms},
                processing_time_ms=processing_time_ms,
                error=str(e)
            )
    
    async def batch_parse(
        self,
        texts: List[str],
        mode: str = "auto",
        correlation_id: Optional[str] = None
    ) -> List[AgentJSONResult]:
        """
        Parse multiple JSON texts in batch.
        
        Args:
            texts: List of texts containing JSON
            mode: Parsing mode
            correlation_id: Correlation ID for tracking
            
        Returns:
            List of AgentJSONResult objects
        """
        correlation_id = correlation_id or f"agentjson_batch_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')}"
        
        start_time = datetime.now(timezone.utc)
        
        logger.info({
            "msg": "Starting AgentJSON batch parsing",
            "text_count": len(texts),
            "mode": mode,
            "correlation_id": correlation_id,
            "timestamp": start_time.isoformat()
        })
        
        try:
            # Process texts in parallel
            tasks = [
                self.parse_json(
                    text=text,
                    mode=mode,
                    correlation_id=f"{correlation_id}_text_{i}"
                )
                for i, text in enumerate(texts)
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Handle any exceptions in the gathered results
            processed_results = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error({
                        "msg": f"Batch item {i} parsing failed",
                        "correlation_id": f"{correlation_id}_text_{i}",
                        "error": str(result)
                    })
                    processed_results.append(AgentJSONResult(
                        success=False,
                        parsed_data=None,
                        status="failed",
                        confidence=0.0,
                        repairs_applied=[],
                        metadata={"batch_index": i, "error": str(result)},
                        error=str(result)
                    ))
                else:
                    processed_results.append(result)
            
            processing_time_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            
            successful_count = sum(1 for r in processed_results if r.success)
            
            logger.info({
                "msg": "AgentJSON batch parsing completed",
                "correlation_id": correlation_id,
                "text_count": len(texts),
                "successful_count": successful_count,
                "processing_time_ms": processing_time_ms,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            
            return processed_results
            
        except Exception as e:
            processing_time_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            
            logger.error({
                "msg": "AgentJSON batch parsing failed",
                "correlation_id": correlation_id,
                "error": str(e),
                "processing_time_ms": processing_time_ms,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            
            # Return error results for all texts
            error_results = []
            for i in range(len(texts)):
                error_results.append(AgentJSONResult(
                    success=False,
                    parsed_data=None,
                    status="failed",
                    confidence=0.0,
                    repairs_applied=[],
                    metadata={"batch_index": i, "error": str(e)},
                    processing_time_ms=processing_time_ms / len(texts) if texts else 0.0,
                    error=str(e)
                ))
            
            return error_results
    
    async def get_repair_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about common JSON repairs.
        
        Returns:
            Dictionary with repair statistics
        """
        stats = {
            "common_repairs": [
                {"type": "unquoted_keys", "description": "Adding quotes to unquoted keys"},
                {"type": "trailing_commas", "description": "Removing trailing commas"},
                {"type": "single_quotes", "description": "Converting single quotes to double"},
                {"type": "python_literals", "description": "Converting Python literals (True/False/None)"},
                {"type": "missing_commas", "description": "Adding missing commas between items"},
                {"type": "unclosed_strings", "description": "Closing unclosed strings"},
                {"type": "markdown_fences", "description": "Stripping markdown code fences"},
                {"type": "prefix_suffix", "description": "Removing prefix/suffix garbage"}
            ],
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        return stats
    
    def get_agentjson_status(self) -> Dict[str, Any]:
        """
        Get the status of the AgentJSON integration.
        
        Returns:
            Dictionary with integration status
        """
        return {
            "available": self.parser is not None,
            "mode": self.config.get("mode", "auto"),
            "top_k": self.config.get("top_k", 5),
            "initialized": self.parser is not None,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    
    async def close(self):
        """Close resources used by the integration."""
        logger.info({
            "msg": "Closing AgentJSON integration resources",
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
        
        # No specific cleanup needed for AgentJSON at the moment
        logger.info({
            "msg": "AgentJSON integration resources closed",
            "timestamp": datetime.now(timezone.utc).isoformat()
        })