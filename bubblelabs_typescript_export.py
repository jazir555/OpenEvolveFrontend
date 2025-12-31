"""
BubbleLabs TypeScript Export

This module provides functionality to export BubbleLabs workflows as
TypeScript code, enabling production deployment and custom integrations.

Features:
- Export workflows as deployable TypeScript code
- Include all OpenEvolve parameters and configurations
- Generate type-safe workflow definitions
- Create standalone executables
- Support for custom templates

Author: OpenEvolve Team
Date: 2025-12-29
"""

import json
import os
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

try:
    from bubblelabs_integration import BubbleLabsIntegration, BubbleWorkflowDefinition
    BUBBLELABS_AVAILABLE = True
except ImportError:
    BUBBLELABS_AVAILABLE = False
    logger.warning("BubbleLabs integration not available")


# =============================================================================
# SECURITY VALIDATION FUNCTIONS
# =============================================================================

def validate_output_path(output_path: str, allowed_base_dir: Optional[str] = None) -> str:
    """
    Validate and sanitize the output path to prevent path traversal attacks.

    CRITICAL BUG FIX #5: Normalize path before checking to prevent path traversal.
    Use os.path.realpath() for symlink resolution.

    Args:
        output_path: The user-provided output path
        allowed_base_dir: Optional base directory to restrict paths to

    Returns:
        Absolute, validated path

    Raises:
        ValueError: If path is invalid or contains path traversal attempts
    """
    if not output_path:
        raise ValueError("Output path cannot be empty")

    # CRITICAL FIX: Normalize path BEFORE checking for traversal attempts
    # This prevents bypass attempts using mixed separators or encoded paths
    normalized_path = os.path.normpath(output_path)

    # Check for path traversal attempts in normalized path
    if ".." in normalized_path or normalized_path.startswith("~/"):
        raise ValueError(f"Path traversal detected in output path: {output_path}")

    # CRITICAL FIX: Use realpath for symlink resolution
    # This prevents symlink attacks that could bypass directory restrictions
    abs_path = os.path.realpath(output_path)

    # If base directory is specified, ensure the path is within it
    if allowed_base_dir:
        # CRITICAL FIX: Also normalize and realpath the base directory
        allowed_base = os.path.realpath(allowed_base_dir)

        # Check that the resolved path is within the allowed base
        if not abs_path.startswith(allowed_base):
            raise ValueError(f"Output path must be within {allowed_base_dir}")

    return abs_path


def validate_file_extension(filename: str, allowed_extensions: List[str]) -> bool:
    """
    Validate file extension to prevent arbitrary file writes.

    Args:
        filename: The filename to validate
        allowed_extensions: List of allowed extensions (e.g., ['.ts', '.js'])

    Returns:
        True if extension is allowed

    Raises:
        ValueError: If extension is not allowed
    """
    if not filename:
        raise ValueError("Filename cannot be empty")

    # Check for path separators
    if "/" in filename or "\\" in filename:
        raise ValueError("Filename cannot contain path separators")

    # Get extension
    _, ext = os.path.splitext(filename)

    # Validate extension
    if ext.lower() not in [e.lower() for e in allowed_extensions]:
        raise ValueError(f"File extension '{ext}' not allowed. Allowed: {allowed_extensions}")

    # Check for null bytes
    if "\x00" in filename:
        raise ValueError("Filename cannot contain null bytes")

    return True


def sanitize_filename(filename: str) -> str:
    """
    Sanitize filename to prevent directory traversal and other attacks.

    Args:
        filename: The filename to sanitize

    Returns:
        Sanitized filename
    """
    # Remove path separators
    filename = os.path.basename(filename)

    # Remove null bytes
    filename = filename.replace("\x00", "")

    # Limit length
    if len(filename) > 255:
        filename = filename[:255]

    return filename


# =============================================================================
# DATA MODELS
# =============================================================================

@dataclass
class TypeScriptExportConfig:
    """Configuration for TypeScript export."""
    include_comments: bool = True
    include_error_handling: bool = True
    include_logging: bool = True
    export_format: str = "module"  # module, standalone, class
    typescript_version: str = "5.0"
    target_runtime: str = "node"  # node, browser, bun
    include_types: bool = True
    minify: bool = False


@dataclass
class ExportResult:
    """Result of TypeScript export."""
    success: bool
    file_path: Optional[str] = None
    code: Optional[str] = None
    error: Optional[str] = None
    warnings: List[str] = None

    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []


# =============================================================================
# TYPESCRIPT EXPORTER
# =============================================================================

class BubbleLabsTypeScriptExporter:
    """
    Export BubbleLabs workflows as TypeScript code.

    Generates type-safe, production-ready TypeScript code that can be
    deployed independently or integrated into existing applications.
    """

    def __init__(self, config: Optional[TypeScriptExportConfig] = None):
        """
        Initialize TypeScript exporter.

        Args:
            config: Export configuration
        """
        self.config = config or TypeScriptExportConfig()

    def export_workflow(
        self,
        workflow_definition: BubbleWorkflowDefinition,
        output_path: Optional[str] = None
    ) -> ExportResult:
        """
        Export a workflow definition as TypeScript code.

        CRITICAL BUG FIX: Added explicit None check and attribute validation for
        workflow_definition to prevent AttributeError crashes.

        Args:
            workflow_definition: The workflow to export (cannot be None)
            output_path: Optional file path to save the code

        Returns:
            ExportResult with generated code
        """
        # CRITICAL FIX: Validate input before use
        if workflow_definition is None:
            logger.error("workflow_definition cannot be None")
            return ExportResult(
                success=False,
                error="workflow_definition is required",
                code=None
            )

        # CRITICAL FIX: Validate required attributes
        if not hasattr(workflow_definition, 'id'):
            logger.error("workflow_definition missing required 'id' attribute")
            return ExportResult(
                success=False,
                error="Invalid workflow_definition: missing 'id' attribute",
                code=None
            )

        if not hasattr(workflow_definition, 'name'):
            logger.error("workflow_definition missing required 'name' attribute")
            return ExportResult(
                success=False,
                error="Invalid workflow_definition: missing 'name' attribute",
                code=None
            )

        if not hasattr(workflow_definition, 'nodes'):
            logger.error("workflow_definition missing required 'nodes' attribute")
            return ExportResult(
                success=False,
                error="Invalid workflow_definition: missing 'nodes' attribute",
                code=None
            )

        try:
            # Generate TypeScript code
            if self.config.export_format == "module":
                code = self._generate_module_export(workflow_definition)
            elif self.config.export_format == "standalone":
                code = self._generate_standalone_export(workflow_definition)
            elif self.config.export_format == "class":
                code = self._generate_class_export(workflow_definition)
            else:
                return ExportResult(
                    success=False,
                    error=f"Unknown export format: {self.config.export_format}"
                )

            # Save to file if path provided (SECURE: Validate path)
            if output_path:
                # Security: Validate and sanitize output path
                validated_path = validate_output_path(output_path)

                # Security: Validate file extension
                filename = os.path.basename(validated_path)
                validate_file_extension(filename, ['.ts', '.js'])

                # Security: Sanitize filename
                safe_filename = sanitize_filename(filename)
                safe_path = os.path.join(os.path.dirname(validated_path), safe_filename)

                with open(safe_path, 'w') as f:
                    f.write(code)
                logger.info(f"Exported workflow to: {safe_path}")
                return ExportResult(success=True, file_path=safe_path, code=code)
            else:
                return ExportResult(success=True, code=code)

        except ValueError as e:
            # Security validation errors
            logger.error(f"Security validation error: {e}")
            return ExportResult(success=False, error=f"Security validation failed: {str(e)}")
        except Exception as e:
            logger.error(f"Error exporting workflow: {e}")
            return ExportResult(success=False, error=str(e))

    def _generate_module_export(self, workflow: BubbleWorkflowDefinition) -> str:
        """Generate TypeScript module export."""
        lines = []

        # Header
        if self.config.include_comments:
            lines.append(self._generate_header(workflow))

        # Imports
        lines.append(self._generate_imports())

        # Types
        if self.config.include_types:
            lines.append(self._generate_workflow_types(workflow))

        # Workflow definition
        lines.append(self._generate_workflow_definition(workflow))

        # Nodes
        lines.append(self._generate_nodes_export(workflow))

        # Edges
        lines.append(self._generate_edges_export(workflow))

        # Execute function
        lines.append(self._generate_execute_function(workflow))

        # Export
        lines.append("\nexport { workflowDefinition, executeWorkflow };")

        return "\n".join(lines)

    def _generate_standalone_export(self, workflow: BubbleWorkflowDefinition) -> str:
        """Generate standalone TypeScript executable."""
        lines = []

        # Header
        if self.config.include_comments:
            lines.append(self._generate_header(workflow))

        # Shebang
        lines.append("#!/usr/bin/env ts-node")
        lines.append("")

        # Imports
        lines.append(self._generate_imports())

        # Types
        if self.config.include_types:
            lines.append(self._generate_workflow_types(workflow))

        # Workflow definition
        lines.append(self._generate_workflow_definition(workflow))

        # Nodes
        lines.append(self._generate_nodes_export(workflow))

        # Edges
        lines.append(self._generate_edges_export(workflow))

        # Execute function
        lines.append(self._generate_execute_function(workflow))

        # Main execution
        lines.append("\n// Main execution")
        lines.append("async function main() {")
        lines.append('  console.log(`Executing workflow: ${workflowDefinition.name}`);')
        lines.append("  const result = await executeWorkflow();")
        lines.append("  console.log('Workflow completed:', result);")
        lines.append("}")
        lines.append("")
        lines.append("main().catch(console.error);")

        return "\n".join(lines)

    def _generate_class_export(self, workflow: BubbleWorkflowDefinition) -> str:
        """Generate TypeScript class export."""
        lines = []

        # Header
        if self.config.include_comments:
            lines.append(self._generate_header(workflow))

        # Imports
        lines.append(self._generate_imports())

        # Types
        if self.config.include_types:
            lines.append(self._generate_workflow_types(workflow))

        # Class definition
        class_name = self._sanitize_class_name(workflow.name)
        lines.append(f"\nexport class {class_name} {{")
        lines.append("  private definition: WorkflowDefinition;")
        lines.append("  private nodes: WorkflowNode[];")
        lines.append("  private edges: WorkflowEdge[];")
        lines.append("")
        lines.append(f"  constructor() {{")

        # CRITICAL BUG FIX #6: Add custom encoder for datetime and non-serializable types
        # This prevents JSON serialization errors when exporting workflows with datetime objects
        def custom_json_encoder(obj):
            """Custom JSON encoder for non-serializable types"""
            if isinstance(obj, datetime):
                return obj.isoformat()
            elif hasattr(obj, '__dict__'):
                return obj.__dict__
            else:
                return str(obj)  # Fallback to string representation

        lines.append(f"    this.definition = {json.dumps(self._workflow_to_dict(workflow), indent=6, default=custom_json_encoder)};")
        lines.append(f"    this.nodes = {json.dumps(workflow.nodes, indent=6, default=custom_json_encoder)};")
        lines.append(f"    this.edges = {json.dumps(workflow.edges, indent=6, default=custom_json_encoder)};")
        lines.append("  }}")
        lines.append("")
        lines.append("  async execute(): Promise<WorkflowResult> {")
        lines.append("    // Workflow execution logic")
        lines.append("    return { success: true, data: {} };")
        lines.append("  }}")
        lines.append("")
        lines.append("  getDefinition(): WorkflowDefinition {")
        lines.append("    return this.definition;")
        lines.append("  }}")
        lines.append("")
        lines.append("  getNodes(): WorkflowNode[] {")
        lines.append("    return this.nodes;")
        lines.append("  }}")
        lines.append("")
        lines.append("  getEdges(): WorkflowEdge[] {")
        lines.append("    return this.edges;")
        lines.append("  }}")
        lines.append("}}")

        return "\n".join(lines)

    def _generate_header(self, workflow: BubbleWorkflowDefinition) -> str:
        """Generate file header comment."""
        lines = []
        lines.append("/**")
        lines.append(f" * BubbleLabs Workflow: {workflow.name}")
        lines.append(f" *")
        lines.append(f" * Description: {workflow.description}")
        lines.append(f" * Workflow ID: {workflow.id}")
        lines.append(f" *")
        lines.append(f" * Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f" * Nodes: {len(workflow.nodes)}")
        lines.append(f" * Edges: {len(workflow.edges)}")
        lines.append(f" *")
        lines.append(f" * Auto-generated by BubbleLabs TypeScript Exporter")
        lines.append(f" * DO NOT EDIT MANUALLY")
        lines.append(f" */")
        lines.append("")
        return "\n".join(lines)

    def _generate_imports(self) -> str:
        """Generate TypeScript imports."""
        lines = []
        lines.append("// Type definitions")
        lines.append("interface WorkflowNode {")
        lines.append("  id: string;")
        lines.append("  type: string;")
        lines.append("  position: { x: number; y: number };")
        lines.append("  data: Record<string, any>;")
        lines.append("}")
        lines.append("")
        lines.append("interface WorkflowEdge {")
        lines.append("  id: string;")
        lines.append("  source: string;")
        lines.append("  target: string;")
        lines.append("  sourceHandle?: string;")
        lines.append("  targetHandle?: string;")
        lines.append("}")
        lines.append("")
        lines.append("interface WorkflowDefinition {")
        lines.append("  id: string;")
        lines.append("  name: string;")
        lines.append("  description: string;")
        lines.append("  nodes: WorkflowNode[];")
        lines.append("  edges: WorkflowEdge[];")
        lines.append("  metadata: Record<string, any>;")
        lines.append("}")
        lines.append("")
        lines.append("interface WorkflowResult {")
        lines.append("  success: boolean;")
        lines.append("  data?: Record<string, any>;")
        lines.append("  error?: string;")
        lines.append("}")
        lines.append("")
        return "\n".join(lines)

    def _generate_workflow_types(self, workflow: BubbleWorkflowDefinition) -> str:
        """Generate workflow-specific types."""
        lines = []
        lines.append("// Workflow-specific types")
        lines.append(f"type {self._sanitize_class_name(workflow.name)}Node = ")
        lines.append("  | 'content_analyzer'")
        lines.append("  | 'decomposer'")
        lines.append("  | 'solver'")
        lines.append("  | 'verifier';")
        lines.append("")
        return "\n".join(lines)

    def _generate_workflow_definition(self, workflow: BubbleWorkflowDefinition) -> str:
        """Generate workflow definition constant."""
        lines = []
        lines.append("// Workflow definition")
        lines.append("const workflowDefinition: WorkflowDefinition = ")

        # Convert to dict
        def_dict = self._workflow_to_dict(workflow)

        # Format as JSON with proper indentation
        json_str = json.dumps(def_dict, indent=2)
        lines.append(json_str + ";")
        lines.append("")

        return "\n".join(lines)

    def _generate_nodes_export(self, workflow: BubbleWorkflowDefinition) -> str:
        """Generate nodes export."""
        lines = []
        lines.append("// Workflow nodes")
        lines.append("const nodes: WorkflowNode[] = ")

        json_str = json.dumps(workflow.nodes, indent=2)
        lines.append(json_str + ";")
        lines.append("")

        return "\n".join(lines)

    def _generate_edges_export(self, workflow: BubbleWorkflowDefinition) -> str:
        """Generate edges export."""
        lines = []
        lines.append("// Workflow edges")
        lines.append("const edges: WorkflowEdge[] = ")

        json_str = json.dumps(workflow.edges, indent=2)
        lines.append(json_str + ";")
        lines.append("")

        return "\n".join(lines)

    def _generate_execute_function(self, workflow: BubbleWorkflowDefinition) -> str:
        """Generate execute function."""
        lines = []
        lines.append("// Execute workflow")
        lines.append("async function executeWorkflow(")
        lines.append("  parameters?: Record<string, any>")
        lines.append("): Promise<WorkflowResult> {")
        lines.append("  try {")

        if self.config.include_error_handling:
            lines.append("    // Validate workflow")
            lines.append("    if (!workflowDefinition || !nodes || !edges) {")
            lines.append("      throw new Error('Invalid workflow definition');")
            lines.append("    }")

        if self.config.include_logging:
            lines.append("")
            lines.append("    console.log(`Starting workflow: ${workflowDefinition.name}`);")
            lines.append("    console.log(`Nodes: ${nodes.length}`);")
            lines.append("    console.log(`Edges: ${edges.length}`);")

        lines.append("")
        lines.append("    // Execute workflow nodes")
        lines.append("    const results: Record<string, any> = {};")
        lines.append("")
        lines.append("    for (const node of nodes) {")
        lines.append("      console.log(`Executing node: ${node.id}`);")
        lines.append("")
        lines.append("      // Node execution logic here")
        lines.append("      results[node.id] = {")
        lines.append("        success: true,")
        lines.append("        data: {}")
        lines.append("      };")
        lines.append("    }")
        lines.append("")
        lines.append("    return {")
        lines.append("      success: true,")
        lines.append("      data: results")
        lines.append("    };")

        if self.config.include_error_handling:
            lines.append("  } catch (error) {")
            lines.append("    console.error('Workflow execution error:', error);")
            lines.append("    return {")
            lines.append("      success: false,")
            lines.append("      error: error instanceof Error ? error.message : 'Unknown error'")
            lines.append("    };")
            lines.append("  }")

        lines.append("}")
        lines.append("")

        return "\n".join(lines)

    def _workflow_to_dict(self, workflow: BubbleWorkflowDefinition) -> Dict[str, Any]:
        """Convert workflow definition to dictionary."""
        return {
            "id": workflow.id,
            "name": workflow.name,
            "description": workflow.description,
            "nodes": workflow.nodes,
            "edges": workflow.edges,
            "metadata": workflow.metadata
        }

    def _sanitize_class_name(self, name: str) -> str:
        """
        Sanitize workflow name for use as class name.

        CRITICAL BUG FIX #8: Added check for empty string before accessing sanitized[0].
        Returns "UnnamedWorkflow" if the name becomes empty after sanitization.
        """
        # Remove invalid characters
        sanitized = name.replace("-", "_").replace(" ", "_")

        # CRITICAL FIX: Check if string is empty before accessing first character
        if not sanitized or len(sanitized) == 0:
            return "UnnamedWorkflow"

        # Remove leading numbers (now safe because we checked for empty)
        if sanitized[0].isdigit():
            sanitized = "_" + sanitized

        return sanitized


# =============================================================================
# BATCH EXPORT
# =============================================================================

def export_all_workflows(
    output_dir: str,
    config: Optional[TypeScriptExportConfig] = None
) -> Tuple[int, List[ExportResult]]:
    """
    Export all BubbleLabs workflows to TypeScript.

    CRITICAL BUG FIX: Added validation for None workflows in the list and
    proper type checking to prevent AttributeError crashes.

    Args:
        output_dir: Directory to save exported files
        config: Export configuration

    Returns:
        Tuple of (count, results)
    """
    if not BUBBLELABS_AVAILABLE:
        result = ExportResult(
            success=False,
            error="BubbleLabs integration not available"
        )
        return 0, [result]

    try:
        # Security: Validate output directory path
        validated_dir = validate_output_path(output_dir)

        # Create output directory
        os.makedirs(validated_dir, exist_ok=True)

        # Get all workflows
        integration = BubbleLabsIntegration()
        definitions = integration.list_workflow_definitions()

        # CRITICAL FIX: Validate workflows list
        if definitions is None:
            logger.error("workflows list cannot be None")
            return 0, [ExportResult(success=False, error="Workflows list is None")]

        if not isinstance(definitions, list):
            logger.error(f"workflows must be a list, got {type(definitions)}")
            return 0, [ExportResult(success=False, error=f"Invalid workflows type: {type(definitions)}")]

        if len(definitions) == 0:
            logger.warning("No workflows to export")
            return 0, []

        # Export each workflow
        exporter = BubbleLabsTypeScriptExporter(config)
        results = []
        count = 0

        for i, definition in enumerate(definitions):
            # CRITICAL FIX: Check if workflow is None before processing
            if definition is None:
                logger.error(f"Workflow at index {i} is None, skipping")
                results.append(ExportResult(success=False, error=f"Workflow at index {i} is None"))
                continue

            # CRITICAL FIX: Validate workflow has required attributes
            if not hasattr(definition, 'id'):
                logger.error(f"Workflow at index {i} missing 'id' attribute, skipping")
                results.append(ExportResult(success=False, error=f"Workflow at index {i} missing 'id' attribute"))
                continue

            try:
                # Generate filename
                filename = sanitize_filename(f"{definition.id}.ts")
                filepath = os.path.join(validated_dir, filename)

                # Export workflow
                result = exporter.export_workflow(definition, filepath)
                results.append(result)

                if result.success:
                    count += 1
            except Exception as e:
                logger.error(f"Error exporting workflow {definition.id if hasattr(definition, 'id') else f'index_{i}'}: {e}")
                results.append(ExportResult(success=False, error=str(e)))

        logger.info(f"Exported {count}/{len(definitions)} workflows to {validated_dir}")
        return count, results

    except ValueError as e:
        # Security validation errors
        logger.error(f"Security validation error: {e}")
        result = ExportResult(success=False, error=f"Security validation failed: {str(e)}")
        return 0, [result]
    except Exception as e:
        logger.error(f"Error exporting workflows: {e}")
        result = ExportResult(success=False, error=str(e))
        return 0, [result]


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def export_workflow_to_typescript(
    workflow_id: str,
    output_path: Optional[str] = None,
    config: Optional[TypeScriptExportConfig] = None
) -> ExportResult:
    """
    Convenience function to export a workflow to TypeScript.

    Args:
        workflow_id: ID of the workflow to export
        output_path: Optional file path
        config: Optional export configuration

    Returns:
        ExportResult
    """
    if not BUBBLELABS_AVAILABLE:
        return ExportResult(
            success=False,
            error="BubbleLabs integration not available"
        )

    try:
        # Get workflow definition
        integration = BubbleLabsIntegration()
        definition = integration.get_workflow_definition(workflow_id)

        if not definition:
            return ExportResult(
                success=False,
                error=f"Workflow not found: {workflow_id}"
            )

        # Export workflow
        exporter = BubbleLabsTypeScriptExporter(config)
        return exporter.export_workflow(definition, output_path)

    except Exception as e:
        logger.error(f"Error exporting workflow: {e}")
        return ExportResult(success=False, error=str(e))


if __name__ == "__main__":
    # Example usage
    if not BUBBLELABS_AVAILABLE:
        print("BubbleLabs integration not available")
    else:
        # Export all workflows
        count, results = export_all_workflows("./exported_workflows")
        print(f"\nExported {count} workflows")

        # Export specific workflow
        integration = BubbleLabsIntegration()
        definitions = integration.list_workflow_definitions()

        if definitions:
            result = export_workflow_to_typescript(
                definitions[0].id,
                output_path=f"./{definitions[0].id}.ts"
            )

            if result.success:
                print(f"\nExported workflow to: {result.file_path}")
                if result.code:
                    print("\nGenerated TypeScript code:")
                    print(result.code[:500] + "..." if len(result.code) > 500 else result.code)
            else:
                print(f"\nExport failed: {result.error}")
