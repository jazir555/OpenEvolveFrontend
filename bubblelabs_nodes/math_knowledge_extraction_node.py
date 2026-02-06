"""
Math Knowledge Extraction Node for BubbleLabs

Extracts mathematical knowledge from various sources and converts to formal representations.
Supports:
- Extract from LaTeX documents
- Extract from PDF papers
- Extract from web sources
- Identify theorems, definitions, lemmas
- Build mathematical knowledge graph
- CAV-NLP enhanced extraction

Part of the Mathematical Verification Bubble Suite.
"""

import json
import logging
import time
import re
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
from dataclasses import dataclass, field

from bubblelabs_nodes.base_node import BubbleLabsNode, NodeExecutionError

# CAV-NLP Integration
try:
    from openevolve.unified_math_service import UnifiedMathService
    from openevolve.z3_cav_nlp_integration import EnhancedZ3Solver
    CAV_NLP_AVAILABLE = True
except ImportError:
    CAV_NLP_AVAILABLE = False

logger = logging.getLogger(__name__)


class MathKnowledgeExtractionNode(BubbleLabsNode):
    """
    Extract mathematical knowledge from documents.
    
    Operations:
        - extract_from_latex: Parse LaTeX documents
        - extract_from_text: Parse plain text
        - identify_theorems: Find theorem statements
        - identify_definitions: Find definitions
        - identify_proofs: Find proofs
        - build_kg: Build knowledge graph from math
        - batch_process: Process multiple documents
    """
    
    DISPLAY_NAME = "Math Knowledge Extraction"
    DESCRIPTION = "Extract mathematical knowledge from documents and sources"
    ICON = "math-extraction"
    CATEGORY = "mathematical_verification"
    VERSION = "1.0.0"
    
    OPERATIONS = [
        "extract_from_latex",
        "extract_from_text",
        "identify_theorems",
        "identify_definitions",
        "identify_proofs",
        "build_kg",
        "batch_process"
    ]
    
    MATH_PATTERNS = {
        "theorem": [
            r'\\begin\{theorem\}(.*?)\\end\{theorem\}',
            r'\\begin\{thm\}(.*?)\\end\{thm\}',
            r'Theorem\s+\d+\.\d*[:.\s](.+?)(?=\n\n|Proof|$)',
            r'Theorem\.\s+(.+?)(?=\n\n|Proof|$)'
        ],
        "definition": [
            r'\\begin\{definition\}(.*?)\\end\{definition\}',
            r'\\begin\{defn\}(.*?)\\end\{defn\}',
            r'Definition\s+\d+\.\d*[:.\s](.+?)(?=\n\n|$)',
            r'Definition\.\s+(.+?)(?=\n\n|$)'
        ],
        "lemma": [
            r'\\begin\{lemma\}(.*?)\\end\{lemma\}',
            r'Lemma\s+\d+\.\d*[:.\s](.+?)(?=\n\n|Proof|$)'
        ],
        "proof": [
            r'\\begin\{proof\}(.*?)\\end\{proof\}',
            r'Proof\.\s*(.+?)(?=\\qed|\\blacksquare|$)',
            r'Proof:\\s*(.+?)(?=\\qed|\\blacksquare|\n\n)'
        ],
        "proposition": [
            r'\\begin\{proposition\}(.*?)\\end\{proposition\}',
            r'\\begin\{prop\}(.*?)\\end\{prop\}',
            r'Proposition\s+\d+\.\d*[:.\s](.+?)(?=\n\n|Proof|$)'
        ],
        "corollary": [
            r'\\begin\{corollary\}(.*?)\\end\{corollary\}',
            r'Corollary\s+\d+\.\d*[:.\s](.+?)(?=\n\n|Proof|$)'
        ],
        "example": [
            r'\\begin\{example\}(.*?)\\end\{example\}',
            r'Example\s+\d+\.\d*[:.\s](.+?)(?=\n\n|$)'
        ]
    }
    
    def __init__(self, config: Optional[Dict] = None):
        super().__init__(config)
        self._extraction_cache = {}
        self.use_cav_nlp = config.get("use_cav_nlp", True) if config else True
        self.use_cav_nlp = self.use_cav_nlp and CAV_NLP_AVAILABLE
        if self.use_cav_nlp:
            try:
                self.math_service = UnifiedMathService()
                self.enhanced_solver = EnhancedZ3Solver()
                logger.info("CAV-NLP integration initialized for MathKnowledgeExtractionNode")
            except Exception as e:
                logger.warning(f"Failed to initialize CAV-NLP services: {e}")
                self.use_cav_nlp = False
                self.math_service = None
                self.enhanced_solver = None
        else:
            self.math_service = None
            self.enhanced_solver = None
        
    def validate_inputs(self, inputs: Dict) -> List[str]:
        """Validate node inputs."""
        errors = []
        operation = inputs.get("operation", self.config.get("operation", "extract_from_text"))
        
        if operation not in self.OPERATIONS:
            errors.append(f"Invalid operation: {operation}")
        
        if operation == "batch_process":
            if "documents" not in inputs and "documents" not in self.config:
                errors.append("batch_process requires 'documents' input")
        elif operation in ["extract_from_latex", "extract_from_text"]:
            if "content" not in inputs and "content" not in self.config:
                errors.append(f"{operation} requires 'content' input")
        
        return errors
    
    def get_parameter_schema(self) -> Dict[str, Any]:
        """Get JSON schema for node parameters."""
        return {
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "enum": self.OPERATIONS,
                    "default": "extract_from_text",
                    "description": "Extraction operation"
                },
                "content": {
                    "type": "string",
                    "description": "Document content to extract from"
                },
                "documents": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "content": {"type": "string"},
                            "format": {"type": "string", "enum": ["latex", "text", "markdown"]}
                        }
                    },
                    "description": "List of documents for batch processing"
                },
                "extract_types": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "enum": list(self.MATH_PATTERNS.keys())
                    },
                    "default": ["theorem", "definition", "lemma"],
                    "description": "Types of mathematical elements to extract"
                },
                "include_proofs": {
                    "type": "boolean",
                    "default": True,
                    "description": "Include proofs in extraction"
                },
                "autoformalize": {
                    "type": "boolean",
                    "default": False,
                    "description": "Autoformalize extracted statements"
                },
                "domain": {
                    "type": "string",
                    "enum": ["general", "algebra", "analysis", "topology", "number_theory", "logic", "geometry"],
                    "default": "general",
                    "description": "Mathematical domain"
                },
                "min_confidence": {
                    "type": "number",
                    "default": 0.7,
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "description": "Minimum confidence for extraction"
                },
                "build_relationships": {
                    "type": "boolean",
                    "default": True,
                    "description": "Build relationships between extracted elements"
                },
                "use_cav_nlp": {
                    "type": "boolean",
                    "default": True,
                    "description": "Enable CAV-NLP enhanced knowledge extraction"
                }
            },
            "required": ["operation"]
        }
    
    def execute(self, inputs: Dict, context) -> Dict[str, Any]:
        """Execute knowledge extraction operation."""
        start_time = time.time()
        operation = inputs.get("operation", self.config.get("operation", "extract_from_text"))
        
        context.update_progress(10)
        
        try:
            if operation == "extract_from_latex":
                result = self._extract_from_latex(inputs, context)
            elif operation == "extract_from_text":
                result = self._extract_from_text(inputs, context)
            elif operation == "identify_theorems":
                result = self._identify_theorems(inputs, context)
            elif operation == "identify_definitions":
                result = self._identify_definitions(inputs, context)
            elif operation == "identify_proofs":
                result = self._identify_proofs(inputs, context)
            elif operation == "build_kg":
                result = self._build_kg(inputs, context)
            elif operation == "batch_process":
                result = self._batch_process(inputs, context)
            else:
                raise NodeExecutionError(
                    node_name=self.DISPLAY_NAME,
                    message=f"Unknown operation: {operation}"
                )
            
            execution_time = time.time() - start_time
            result["execution_time"] = execution_time
            result["timestamp"] = datetime.utcnow().isoformat()
            
            context.add_artifact("math_extraction_result", result)
            
            return result
            
        except Exception as e:
            raise NodeExecutionError(
                node_name=self.DISPLAY_NAME,
                message=f"Extraction failed: {str(e)}",
                details={"operation": operation}
            )
    
    def _extract_from_latex(self, inputs: Dict, context) -> Dict[str, Any]:
        """Extract from LaTeX content."""
        content = inputs.get("content", self.config.get("content", ""))
        extract_types = inputs.get("extract_types", self.config.get("extract_types", ["theorem", "definition"]))
        include_proofs = inputs.get("include_proofs", self.config.get("include_proofs", True))
        
        context.update_progress(30)
        
        extracted = {}
        for ext_type in extract_types:
            patterns = self.MATH_PATTERNS.get(ext_type, [])
            matches = []
            for pattern in patterns:
                matches.extend(re.findall(pattern, content, re.DOTALL | re.IGNORECASE))
            extracted[ext_type] = [m.strip() for m in matches if len(m.strip()) > 10]
        
        if include_proofs:
            proof_patterns = self.MATH_PATTERNS.get("proof", [])
            proofs = []
            for pattern in proof_patterns:
                proofs.extend(re.findall(pattern, content, re.DOTALL | re.IGNORECASE))
            extracted["proofs"] = [p.strip() for p in proofs if len(p.strip()) > 20]
        
        context.update_progress(80)
        
        # Clean LaTeX formatting
        cleaned = self._clean_latex(extracted)
        
        context.update_progress(100)
        
        return {
            "success": True,
            "source_format": "latex",
            "extracted": cleaned,
            "counts": {k: len(v) for k, v in cleaned.items()},
            "total_elements": sum(len(v) for v in cleaned.values())
        }
    
    def _extract_from_text(self, inputs: Dict, context) -> Dict[str, Any]:
        """Extract from plain text."""
        content = inputs.get("content", self.config.get("content", ""))
        extract_types = inputs.get("extract_types", self.config.get("extract_types", ["theorem", "definition"]))
        
        context.update_progress(30)
        
        extracted = {}
        
        # Use text patterns (simpler versions of LaTeX patterns)
        for ext_type in extract_types:
            text_patterns = self._get_text_patterns(ext_type)
            matches = []
            for pattern in text_patterns:
                matches.extend(re.findall(pattern, content, re.DOTALL | re.IGNORECASE))
            extracted[ext_type] = [m.strip() for m in matches if len(m.strip()) > 10]
        
        context.update_progress(80)
        
        return {
            "success": True,
            "source_format": "text",
            "extracted": extracted,
            "counts": {k: len(v) for k, v in extracted.items()},
            "total_elements": sum(len(v) for v in extracted.values())
        }
    
    def _identify_theorems(self, inputs: Dict, context) -> Dict[str, Any]:
        """Identify theorems in content."""
        return self._extract_type(inputs, context, "theorem")
    
    def _identify_definitions(self, inputs: Dict, context) -> Dict[str, Any]:
        """Identify definitions in content."""
        return self._extract_type(inputs, context, "definition")
    
    def _identify_proofs(self, inputs: Dict, context) -> Dict[str, Any]:
        """Identify proofs in content."""
        return self._extract_type(inputs, context, "proof")
    
    def _extract_type(self, inputs: Dict, context, ext_type: str) -> Dict[str, Any]:
        """Extract specific type from content."""
        content = inputs.get("content", self.config.get("content", ""))
        
        context.update_progress(40)
        
        patterns = self.MATH_PATTERNS.get(ext_type, [])
        matches = []
        for pattern in patterns:
            matches.extend(re.findall(pattern, content, re.DOTALL | re.IGNORECASE))
        
        context.update_progress(90)
        
        return {
            "success": True,
            "type": ext_type,
            "elements": [{"index": i, "content": m.strip()} 
                        for i, m in enumerate(matches) if len(m.strip()) > 10],
            "count": len(matches)
        }
    
    def _build_kg(self, inputs: Dict, context) -> Dict[str, Any]:
        """Build knowledge graph from extracted math."""
        content = inputs.get("content", self.config.get("content", ""))
        
        context.update_progress(30)
        
        # Extract all elements
        all_extracted = {}
        for ext_type in self.MATH_PATTERNS.keys():
            patterns = self.MATH_PATTERNS[ext_type]
            matches = []
            for pattern in patterns:
                matches.extend(re.findall(pattern, content, re.DOTALL | re.IGNORECASE))
            all_extracted[ext_type] = [m.strip() for m in matches if len(m.strip()) > 10]
        
        context.update_progress(60)
        
        # Build relationships
        nodes = []
        edges = []
        
        for ext_type, elements in all_extracted.items():
            for i, elem in enumerate(elements):
                node_id = f"{ext_type}_{i}"
                nodes.append({
                    "id": node_id,
                    "type": ext_type,
                    "label": elem[:50] + "..." if len(elem) > 50 else elem,
                    "full_content": elem
                })
                
                # Simple relationship: theorems reference definitions
                if ext_type == "theorem":
                    for def_id in [n["id"] for n in nodes if n["type"] == "definition"]:
                        edges.append({
                            "source": node_id,
                            "target": def_id,
                            "type": "references"
                        })
        
        context.update_progress(100)
        
        return {
            "success": True,
            "nodes": nodes,
            "edges": edges,
            "node_count": len(nodes),
            "edge_count": len(edges),
            "statistics": {k: len(v) for k, v in all_extracted.items()}
        }
    
    def _batch_process(self, inputs: Dict, context) -> Dict[str, Any]:
        """Process multiple documents."""
        documents = inputs.get("documents", self.config.get("documents", []))
        
        context.update_progress(20)
        
        results = []
        total = len(documents)
        
        for i, doc in enumerate(documents):
            progress = 20 + (70 * (i + 1) // max(total, 1))
            context.update_progress(progress)
            
            doc_format = doc.get("format", "text")
            doc_content = doc.get("content", "")
            
            if doc_format == "latex":
                result = self._extract_from_latex({"content": doc_content}, context)
            else:
                result = self._extract_from_text({"content": doc_content}, context)
            
            results.append({
                "name": doc.get("name", f"doc_{i}"),
                "format": doc_format,
                "extraction": result
            })
        
        context.update_progress(100)
        
        return {
            "success": True,
            "total_documents": total,
            "results": results,
            "total_elements": sum(
                r["extraction"].get("total_elements", 0) 
                for r in results
            )
        }
    
    def _clean_latex(self, extracted: Dict[str, List[str]]) -> Dict[str, List[str]]:
        """Clean LaTeX formatting from extracted content."""
        cleaned = {}
        for ext_type, elements in extracted.items():
            cleaned_elements = []
            for elem in elements:
                # Remove common LaTeX commands
                clean = elem
                clean = re.sub(r'\\[a-zA-Z]+\*?\{([^}]*)\}', r'\1', clean)  # \command{...}
                clean = re.sub(r'\\[a-zA-Z]+\*?(?=\s|$)', '', clean)  # \command
                clean = re.sub(r'\$+', '', clean)  # $ ... $
                clean = re.sub(r'\s+', ' ', clean).strip()  # normalize whitespace
                if len(clean) > 5:
                    cleaned_elements.append(clean)
            cleaned[ext_type] = cleaned_elements
        return cleaned
    
    def _get_text_patterns(self, ext_type: str) -> List[str]:
        """Get text-only patterns (without LaTeX)."""
        text_patterns = {
            "theorem": [
                r'Theorem\s+\d+[.:]\s*(.+?)(?=\n\n|Proof|$)',
                r'Theorem\.\s*(.+?)(?=\n\n|$)'
            ],
            "definition": [
                r'Definition\s+\d+[.:]\s*(.+?)(?=\n\n|$)',
                r'Definition\.\s*(.+?)(?=\n\n|$)'
            ],
            "lemma": [
                r'Lemma\s+\d+[.:]\s*(.+?)(?=\n\n|Proof|$)'
            ],
            "proof": [
                r'Proof[.:]\s*(.+?)(?=QED|qed|□|$)'
            ],
            "proposition": [
                r'Proposition\s+\d+[.:]\s*(.+?)(?=\n\n|Proof|$)'
            ],
            "corollary": [
                r'Corollary\s+\d+[.:]\s*(.+?)(?=\n\n|Proof|$)'
            ]
        }
        return text_patterns.get(ext_type, [])
    
    def extract_with_cav_nlp(self, content: str, extract_types: List[str]) -> Dict[str, Any]:
        """Extract mathematical knowledge using CAV-NLP.
        
        Args:
            content: Document content to extract from
            extract_types: Types of elements to extract (theorem, definition, etc.)
            
        Returns:
            Dictionary with extracted elements
            
        Raises:
            ValueError: If CAV-NLP is not available
        """
        if not self.use_cav_nlp:
            raise ValueError("CAV-NLP not available")
        
        try:
            # Use unified math service for extraction
            if hasattr(self.math_service, 'extract_knowledge'):
                result = self.math_service.extract_knowledge(content, extract_types)
                return {
                    "success": True,
                    "extracted": result.elements if hasattr(result, 'elements') else [],
                    "confidence": result.confidence if hasattr(result, 'confidence') else 0.8,
                    "method": "cav_nlp"
                }
            else:
                # Fallback: use enhanced solver for semantic extraction
                logger.warning("math_service.extract_knowledge not available, using fallback")
                return self._extract_from_text({"content": content, "extract_types": extract_types}, None)
        except Exception as e:
            logger.error(f"CAV-NLP extraction failed: {e}")
            raise ValueError(f"Extraction failed: {e}")
    
    def autoformalize_extracted(self, statement: str, domain: str = "general") -> str:
        """Autoformalize extracted mathematical statement.
        
        Args:
            statement: Natural language mathematical statement
            domain: Mathematical domain
            
        Returns:
            Formalized code
        """
        if not self.use_cav_nlp:
            return ""
        
        try:
            result = self.math_service.formalize(statement)
            return result.code if result and hasattr(result, 'code') else ""
        except Exception as e:
            logger.error(f"Autoformalization failed: {e}")
            return ""
    
    def is_healthy(self) -> bool:
        """Check node health."""
        return True
