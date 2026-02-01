"""
Math Proof Simplification Node for BubbleLabs

Simplifies and optimizes mathematical proofs:
- Remove redundant steps
- Combine similar tactics
- Shorten proof sequences
- Improve readability
- Compress proof size

Part of the Mathematical Verification Bubble Suite.
"""

import json
import logging
import re
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime

from bubblelabs_nodes.base_node import BubbleLabsNode, NodeExecutionError

logger = logging.getLogger(__name__)


class MathProofSimplificationNode(BubbleLabsNode):
    """
    Simplify and optimize mathematical proofs.
    
    Operations:
        - simplify: General proof simplification
        - remove_redundancy: Remove redundant steps
        - compress: Compress proof size
        - beautify: Improve readability
        - optimize_tactics: Optimize tactic selection
        - suggest_shortcuts: Suggest proof shortcuts
        - batch_simplify: Simplify multiple proofs
    """
    
    DISPLAY_NAME = "Math Proof Simplification"
    DESCRIPTION = "Simplify and optimize mathematical proofs"
    ICON = "math-simplify"
    CATEGORY = "mathematical_verification"
    VERSION = "1.0.0"
    
    OPERATIONS = [
        "simplify",
        "remove_redundancy",
        "compress",
        "beautify",
        "optimize_tactics",
        "suggest_shortcuts",
        "batch_simplify"
    ]
    
    # Tactic simplification rules
    SIMPLIFICATION_RULES = [
        # Redundant combinations
        (r'intro\s+\w+\s*\n\s*intro\s+\w+', r'intros \1 \2'),
        (r'simp\s*\n\s*simp', r'simp'),
        (r'rw\s+\[([^\]]+)\]\s*\n\s*rw\s+\[\1\]', r'rw [\1]'),
        (r'apply\s+(\w+)\s*\n\s*apply\s+\1', r'apply \1'),
        
        # Unnecessary steps
        (r'simp\s*\n\s*done', r'simp'),
        (r'trivial\s*\n\s*done', r'trivial'),
        
        # Better alternatives
        (r'repeat\s*\{\s*rw\s+\[([^\]]+)\]\s*\}', r'simp only [\1]'),
        (r'rw\s+\[([^\]]+)\]\s*\n\s*simp', r'simp [\1]'),
    ]
    
    # Tactic replacement suggestions
    TACTIC_REPLACEMENTS = {
        "apply and.intro; apply h1; apply h2": "exact ⟨h1, h2⟩",
        "apply or.inl; apply h": "left; exact h",
        "apply or.inr; apply h": "right; exact h",
        "intro h; exfalso; apply h": "contradiction",
        "by_cases h": "by_cases h : _",
    }
    
    def __init__(self, config: Optional[Dict] = None):
        super().__init__(config)
    
    def validate_inputs(self, inputs: Dict) -> List[str]:
        """Validate node inputs."""
        errors = []
        operation = inputs.get("operation", self.config.get("operation", "simplify"))
        
        if operation not in self.OPERATIONS:
            errors.append(f"Invalid operation: {operation}")
        
        if operation == "batch_simplify":
            if "proofs" not in inputs and "proofs" not in self.config:
                errors.append("batch_simplify requires 'proofs' input")
        else:
            if "proof" not in inputs and "proof" not in self.config:
                errors.append(f"{operation} requires 'proof' input")
        
        return errors
    
    def get_parameter_schema(self) -> Dict[str, Any]:
        """Get JSON schema for node parameters."""
        return {
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "enum": self.OPERATIONS,
                    "default": "simplify",
                    "description": "Simplification operation"
                },
                "proof": {
                    "type": "string",
                    "description": "Proof code to simplify"
                },
                "proofs": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of proofs for batch processing"
                },
                "aggressive": {
                    "type": "boolean",
                    "default": False,
                    "description": "Apply aggressive simplification"
                },
                "preserve_comments": {
                    "type": "boolean",
                    "default": True,
                    "description": "Preserve comments in proof"
                },
                "target_reduction": {
                    "type": "number",
                    "default": 0.3,
                    "minimum": 0.0,
                    "maximum": 0.9,
                    "description": "Target size reduction ratio"
                }
            },
            "required": ["operation"]
        }
    
    def execute(self, inputs: Dict, context) -> Dict[str, Any]:
        """Execute simplification operation."""
        operation = inputs.get("operation", self.config.get("operation", "simplify"))
        
        try:
            if operation == "simplify":
                result = self._simplify(inputs, context)
            elif operation == "remove_redundancy":
                result = self._remove_redundancy(inputs, context)
            elif operation == "compress":
                result = self._compress(inputs, context)
            elif operation == "beautify":
                result = self._beautify(inputs, context)
            elif operation == "optimize_tactics":
                result = self._optimize_tactics(inputs, context)
            elif operation == "suggest_shortcuts":
                result = self._suggest_shortcuts(inputs, context)
            elif operation == "batch_simplify":
                result = self._batch_simplify(inputs, context)
            else:
                raise NodeExecutionError(
                    node_name=self.DISPLAY_NAME,
                    message=f"Unknown operation: {operation}"
                )
            
            result["timestamp"] = datetime.utcnow().isoformat()
            context.add_artifact("simplification_result", result)
            
            return result
            
        except Exception as e:
            raise NodeExecutionError(
                node_name=self.DISPLAY_NAME,
                message=f"Simplification failed: {str(e)}",
                details={"operation": operation}
            )
    
    def _simplify(self, inputs: Dict, context) -> Dict[str, Any]:
        """General proof simplification."""
        proof = inputs.get("proof", self.config.get("proof", ""))
        aggressive = inputs.get("aggressive", self.config.get("aggressive", False))
        preserve_comments = inputs.get("preserve_comments", self.config.get("preserve_comments", True))
        
        context.update_progress(30)
        
        original_size = len(proof)
        
        # Extract comments if preserving
        comments = {}
        if preserve_comments:
            proof, comments = self._extract_comments(proof)
        
        context.update_progress(50)
        
        # Apply simplification rules
        simplified = self._apply_simplification_rules(proof)
        
        context.update_progress(70)
        
        # Optimize tactics
        simplified = self._optimize_proof_tactics(simplified, aggressive)
        
        context.update_progress(90)
        
        # Reinsert comments
        if preserve_comments:
            simplified = self._reinsert_comments(simplified, comments)
        
        context.update_progress(100)
        
        new_size = len(simplified)
        reduction = (original_size - new_size) / original_size if original_size > 0 else 0
        
        return {
            "success": True,
            "original": proof[:500] + "..." if len(proof) > 500 else proof,
            "simplified": simplified[:500] + "..." if len(simplified) > 500 else simplified,
            "original_size": original_size,
            "simplified_size": new_size,
            "reduction_ratio": round(reduction, 3),
            "reduction_percent": round(reduction * 100, 1),
            "changes_made": self._count_changes(proof, simplified)
        }
    
    def _remove_redundancy(self, inputs: Dict, context) -> Dict[str, Any]:
        """Remove redundant steps from proof."""
        proof = inputs.get("proof", self.config.get("proof", ""))
        
        context.update_progress(50)
        
        lines = proof.split('\n')
        cleaned = []
        seen = set()
        
        for line in lines:
            line_stripped = line.strip()
            # Skip empty lines
            if not line_stripped:
                cleaned.append(line)
                continue
            # Skip duplicate consecutive identical tactics
            if line_stripped in seen and cleaned and cleaned[-1].strip() == line_stripped:
                continue
            cleaned.append(line)
            seen.add(line_stripped)
        
        result = '\n'.join(cleaned)
        
        context.update_progress(100)
        
        return {
            "success": True,
            "original_lines": len(lines),
            "cleaned_lines": len(cleaned),
            "redundant_removed": len(lines) - len(cleaned),
            "result": result
        }
    
    def _compress(self, inputs: Dict, context) -> Dict[str, Any]:
        """Compress proof size."""
        proof = inputs.get("proof", self.config.get("proof", ""))
        target = inputs.get("target_reduction", self.config.get("target_reduction", 0.3))
        
        context.update_progress(50)
        
        # Apply compression
        compressed = proof
        
        # Remove extra whitespace
        compressed = re.sub(r'\n\s*\n\s*\n', '\n\n', compressed)
        
        # Shorten obvious patterns
        compressed = re.sub(r'by\s+\{\s*simp\s*\}', 'by simp', compressed)
        compressed = re.sub(r'by\s+\{\s*trivial\s*\}', 'by trivial', compressed)
        
        context.update_progress(100)
        
        original_size = len(proof)
        new_size = len(compressed)
        reduction = (original_size - new_size) / original_size
        
        return {
            "success": True,
            "compressed": compressed,
            "original_size": original_size,
            "compressed_size": new_size,
            "reduction": round(reduction, 3),
            "target_achieved": reduction >= target
        }
    
    def _beautify(self, inputs: Dict, context) -> Dict[str, Any]:
        """Improve proof readability."""
        proof = inputs.get("proof", self.config.get("proof", ""))
        
        context.update_progress(50)
        
        # Add consistent indentation
        lines = proof.split('\n')
        beautified_lines = []
        indent = 0
        
        for line in lines:
            stripped = line.strip()
            
            # Decrease indent before closing braces
            if stripped.startswith('}'):
                indent = max(0, indent - 2)
            
            beautified_lines.append('  ' * indent + stripped)
            
            # Increase indent after opening braces
            if stripped.endswith('{') and not stripped.startswith('by'):
                indent += 2
        
        result = '\n'.join(beautified_lines)
        
        context.update_progress(100)
        
        return {
            "success": True,
            "beautified": result,
            "improvements": ["Consistent indentation", "Proper line breaks"]
        }
    
    def _optimize_tactics(self, inputs: Dict, context) -> Dict[str, Any]:
        """Optimize tactic selection."""
        proof = inputs.get("proof", self.config.get("proof", ""))
        
        context.update_progress(50)
        
        optimizations = []
        optimized = proof
        
        # Apply tactic replacements
        for old, new in self.TACTIC_REPLACEMENTS.items():
            if old in optimized:
                optimized = optimized.replace(old, new)
                optimizations.append(f"Replaced '{old}' with '{new}'")
        
        context.update_progress(100)
        
        return {
            "success": True,
            "optimized": optimized,
            "optimizations": optimizations,
            "optimization_count": len(optimizations)
        }
    
    def _suggest_shortcuts(self, inputs: Dict, context) -> Dict[str, Any]:
        """Suggest proof shortcuts."""
        proof = inputs.get("proof", self.config.get("proof", ""))
        
        context.update_progress(50)
        
        suggestions = []
        
        # Pattern-based suggestions
        patterns = [
            (r'intro\s+(\w+)\s*\n\s*intro\s+(\w+)', r'Use "intros \1 \2" instead'),
            (r'simp\s*\n\s*simp', r'Repeated simp is redundant'),
            (r'rw\s+\[([^\]]+)\]\s*\n\s*rw\s+\[([^\]]+)\]', r'Consider "rw [\1, \2]" to combine'),
            (r'apply\s+and\.intro\s*\n\s*apply\s+(\w+)\s*\n\s*apply\s+(\w+)', r'Use "exact ⟨\1, \2⟩" instead'),
        ]
        
        for pattern, suggestion in patterns:
            if re.search(pattern, proof):
                suggestions.append(suggestion)
        
        context.update_progress(100)
        
        return {
            "success": True,
            "suggestions": suggestions,
            "suggestion_count": len(suggestions),
            "potential_savings": f"~{len(suggestions) * 5} lines"
        }
    
    def _batch_simplify(self, inputs: Dict, context) -> Dict[str, Any]:
        """Simplify multiple proofs."""
        proofs = inputs.get("proofs", self.config.get("proofs", []))
        
        results = []
        total = len(proofs)
        
        for i, proof in enumerate(proofs):
            progress = (i + 1) / total * 100
            context.update_progress(progress)
            
            result = self._simplify({"proof": proof}, context)
            results.append({
                "original_size": result.get("original_size", 0),
                "simplified_size": result.get("simplified_size", 0),
                "reduction": result.get("reduction_ratio", 0)
            })
        
        avg_reduction = sum(r["reduction"] for r in results) / len(results) if results else 0
        
        return {
            "success": True,
            "total": total,
            "results": results,
            "average_reduction": round(avg_reduction, 3)
        }
    
    def _extract_comments(self, proof: str) -> Tuple[str, Dict[int, str]]:
        """Extract comments from proof."""
        comments = {}
        lines = proof.split('\n')
        clean_lines = []
        
        for i, line in enumerate(lines):
            if '--' in line:
                parts = line.split('--', 1)
                clean_lines.append(parts[0])
                comments[i] = '--' + parts[1]
            elif '/-' in line and '-/' in line:
                # Block comment - remove for now
                clean_lines.append(re.sub(r'/-.*?-/', '', line))
            else:
                clean_lines.append(line)
        
        return '\n'.join(clean_lines), comments
    
    def _reinsert_comments(self, proof: str, comments: Dict[int, str]) -> str:
        """Reinsert comments into proof."""
        lines = proof.split('\n')
        result = []
        
        for i, line in enumerate(lines):
            if i in comments:
                line = line + ' ' + comments[i]
            result.append(line)
        
        return '\n'.join(result)
    
    def _apply_simplification_rules(self, proof: str) -> str:
        """Apply simplification rules to proof."""
        simplified = proof
        for pattern, replacement in self.SIMPLIFICATION_RULES:
            simplified = re.sub(pattern, replacement, simplified)
        return simplified
    
    def _optimize_proof_tactics(self, proof: str, aggressive: bool) -> str:
        """Optimize tactics in proof."""
        optimized = proof
        
        # Basic optimizations
        optimized = re.sub(r'by\s+\{\s*simp\s*\}', 'by simp', optimized)
        optimized = re.sub(r'by\s+\{\s*trivial\s*\}', 'by trivial', optimized)
        
        if aggressive:
            # More aggressive optimizations
            optimized = re.sub(r'simp\s*\n\s*simp', 'simp', optimized)
        
        return optimized
    
    def _count_changes(self, original: str, simplified: str) -> int:
        """Count number of changes made."""
        orig_lines = original.split('\n')
        simp_lines = simplified.split('\n')
        
        # Simple line count difference
        return abs(len(orig_lines) - len(simp_lines))
    
    def is_healthy(self) -> bool:
        """Check node health."""
        return True
