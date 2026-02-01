"""
Knowledge Import/Export Node for BubbleLabs

Provides comprehensive import/export operations for knowledge graphs in multiple formats:
- JSON: Native knowledge graph format
- RDF/TTL: Linked data format (Turtle)
- CSV: Tabular format for entities/relationships
- N-Quads: RDF with context
- NetworkX: Python graph format

Features:
- Export knowledge to various formats
- Import knowledge from external sources
- Transform between formats
- Validate during import
- Handle incremental updates with merge strategies
- Compression support (gzip, zip)
- Progress tracking
"""

import json
import csv
import gzip
import zipfile
import io
import os
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse
import urllib.request
from .base_node import BubbleLabsNode, NodeExecutionError


class KnowledgeImportExportNode(BubbleLabsNode):
    """
    Import and export knowledge in multiple formats (JSON, RDF, CSV, TTL, N-Quads, NetworkX).

    Supports:
    - Export: Save knowledge to files in various formats
    - Import: Load knowledge from external sources
    - Transform: Convert between formats without storage
    - Validation: Validate data integrity during import
    - Incremental updates: Merge strategies for existing data
    - Compression: gzip and zip support for exports
    """

    # Node metadata
    DISPLAY_NAME = "Import/Export"
    DESCRIPTION = "Import and export knowledge in multiple formats (JSON, RDF, CSV, TTL)"
    ICON = "import-export"
    CATEGORY = "knowledge"
    VERSION = "1.0.0"

    # Constants
    MAX_FILE_SIZE_MB = 500  # Maximum file size in MB
    SUPPORTED_FORMATS = ["json", "rdf", "ttl", "csv", "nquads", "networkx"]
    SUPPORTED_OPERATIONS = ["export", "import", "transform"]
    SUPPORTED_COMPRESSION = ["none", "gzip", "zip"]
    SUPPORTED_MERGE_STRATEGIES = ["replace", "merge", "skip_existing"]

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)

        # Safe imports from knowledge_engine
        unified_hub_module = self.safe_import(
            'knowledge_engine.unified_kg_integration_hub',
            fallback_value=None,
            error_msg="Knowledge Engine (unified_kg_integration_hub) not available"
        )

        self.UnifiedKGIntegrationHub = None
        self.UnifiedKGConfig = None
        self.KnowledgeTriple = None
        self.KGSource = None

        if unified_hub_module:
            self.UnifiedKGIntegrationHub = getattr(unified_hub_module, 'UnifiedKGIntegrationHub', None)
            self.UnifiedKGConfig = getattr(unified_hub_module, 'UnifiedKGConfig', None)
            self.KnowledgeTriple = getattr(unified_hub_module, 'KnowledgeTriple', None)
            self.KGSource = getattr(unified_hub_module, 'KGSource', None)

        # Safe import of ExportImportManager
        export_import_module = self.safe_import(
            'export_import_manager',
            fallback_value=None,
            error_msg="ExportImportManager not available"
        )
        self.ExportImportManager = None
        if export_import_module:
            self.ExportImportManager = getattr(export_import_module, 'ExportImportManager', None)

        # Initialize hub instance
        self.hub = None
        self._hub_initialized = False
        if self.UnifiedKGIntegrationHub and self.UnifiedKGConfig:
            try:
                config_obj = self.UnifiedKGConfig()
                self.hub = self.UnifiedKGIntegrationHub(config=config_obj)
                self.logger.info("UnifiedKGIntegrationHub initialized successfully")
            except Exception as e:
                self.logger.warning(f"Could not initialize UnifiedKGIntegrationHub: {e}")
                self.hub = None

    def validate_inputs(self, inputs: Dict) -> List[str]:
        """
        Validate input parameters based on operation type.

        Required fields depend on operation:
        - export: destination_path (optional, uses config)
        - import: source_path (required)
        - transform: source_path and destination_path (required)
        """
        errors = []

        # Check for operation type in inputs or config
        operation = inputs.get('operation', self.config.get('operation', 'export'))

        if operation not in self.SUPPORTED_OPERATIONS:
            errors.append(
                f"Invalid operation: {operation}. "
                f"Must be one of: {', '.join(self.SUPPORTED_OPERATIONS)}"
            )

        # Validate format
        fmt = inputs.get('format', self.config.get('format', 'json'))
        if fmt not in self.SUPPORTED_FORMATS:
            errors.append(
                f"Invalid format: {fmt}. "
                f"Must be one of: {', '.join(self.SUPPORTED_FORMATS)}"
            )

        # Operation-specific validation
        if operation == 'import':
            source = inputs.get('source_path', self.config.get('source_path'))
            if not source:
                errors.append("Missing required field 'source_path' for import operation")

        elif operation == 'export':
            # destination_path is optional for export (can return data directly)
            pass

        elif operation == 'transform':
            source = inputs.get('source_path', self.config.get('source_path'))
            dest = inputs.get('destination_path', self.config.get('destination_path'))
            if not source:
                errors.append("Missing required field 'source_path' for transform operation")
            if not dest:
                errors.append("Missing required field 'destination_path' for transform operation")

        # Validate entity_types if provided
        if 'entity_types' in inputs:
            if not isinstance(inputs['entity_types'], list):
                errors.append("'entity_types' must be an array of strings")
            else:
                for i, etype in enumerate(inputs['entity_types']):
                    if not isinstance(etype, str):
                        errors.append(f"'entity_types[{i}]' must be a string")

        # Validate merge_strategy if provided
        if 'merge_strategy' in inputs:
            strategy = inputs['merge_strategy']
            if strategy not in self.SUPPORTED_MERGE_STRATEGIES:
                errors.append(
                    f"Invalid merge_strategy: {strategy}. "
                    f"Must be one of: {', '.join(self.SUPPORTED_MERGE_STRATEGIES)}"
                )

        # Validate compression if provided
        if 'compression' in inputs:
            compression = inputs['compression']
            if compression not in self.SUPPORTED_COMPRESSION:
                errors.append(
                    f"Invalid compression: {compression}. "
                    f"Must be one of: {', '.join(self.SUPPORTED_COMPRESSION)}"
                )

        return errors

    def execute(self, inputs: Dict, context) -> Dict[str, Any]:
        """
        Execute the import/export operation based on configuration.

        Args:
            inputs: Input data containing operation and operation-specific parameters
            context: Workflow state for tracking progress

        Returns:
            Dict containing:
                - success: Boolean indicating operation success
                - records_processed: Number of records processed
                - file_path: Path to exported file (for export operations)
                - data: Exported data (when no destination_path specified)
                - errors: List of error messages

        Raises:
            NodeExecutionError: If execution fails
        """
        operation = inputs.get('operation', self.config.get('operation', 'export'))
        fmt = inputs.get('format', self.config.get('format', 'json'))

        context.update_progress(10, f"Starting {operation} operation with format {fmt}")
        self.logger.info(f"Executing {operation} with format {fmt}")

        try:
            if operation == 'export':
                result = self._execute_export(inputs, context)
            elif operation == 'import':
                result = self._execute_import(inputs, context)
            elif operation == 'transform':
                result = self._execute_transform(inputs, context)
            else:
                raise NodeExecutionError(
                    node_name=self.get_display_name(),
                    message=f"Unknown operation: {operation}",
                    details={'valid_operations': self.SUPPORTED_OPERATIONS}
                )

            context.update_progress(100, f"{operation.capitalize()} operation completed")

            # Add artifact to context
            context.add_artifact('knowledge_import_export', {
                'operation': operation,
                'format': fmt,
                'success': result.get('success', False),
                'records_processed': result.get('records_processed', 0)
            })

            return result

        except NodeExecutionError:
            raise
        except Exception as e:
            self.logger.error(f"{operation} operation failed: {e}", exc_info=True)
            raise NodeExecutionError(
                node_name=self.get_display_name(),
                message=f"{operation.capitalize()} operation failed: {str(e)}",
                details={
                    'operation': operation,
                    'format': fmt,
                    'inputs': inputs,
                    'exception_type': type(e).__name__
                }
            ) from e

    def _execute_export(self, inputs: Dict, context) -> Dict[str, Any]:
        """Export knowledge to file or return as data."""
        fmt = inputs.get('format', self.config.get('format', 'json'))
        destination = inputs.get('destination_path', self.config.get('destination_path'))
        compression = inputs.get('compression', self.config.get('compression', 'none'))
        include_metadata = inputs.get('include_metadata', self.config.get('include_metadata', True))
        entity_types = inputs.get('entity_types', self.config.get('entity_types', []))

        context.update_progress(20, "Preparing knowledge for export")

        # Get knowledge data from hub or inputs
        if self.hub:
            knowledge_data = self._get_knowledge_from_hub(entity_types)
        else:
            knowledge_data = inputs.get('knowledge_data', {
                'entities': {},
                'relations': {},
                'triples': []
            })

        context.update_progress(40, f"Converting to {fmt} format")

        # Convert to target format
        try:
            if fmt == 'json':
                export_data = self._export_to_json(knowledge_data, include_metadata)
            elif fmt in ['rdf', 'ttl']:
                export_data = self._export_to_ttl(knowledge_data, include_metadata)
            elif fmt == 'csv':
                export_data = self._export_to_csv(knowledge_data)
            elif fmt == 'nquads':
                export_data = self._export_to_nquads(knowledge_data)
            elif fmt == 'networkx':
                export_data = self._export_to_networkx(knowledge_data)
            else:
                raise ValueError(f"Unsupported format: {fmt}")
        except Exception as e:
            return {
                'success': False,
                'records_processed': 0,
                'file_path': None,
                'errors': [f"Format conversion failed: {str(e)}"]
            }

        context.update_progress(70, "Applying compression if requested")

        # Apply compression if requested
        if compression != 'none' and destination:
            export_data = self._apply_compression(export_data, compression)
            destination = self._update_extension_for_compression(destination, compression)

        context.update_progress(90, "Writing to destination")

        # Write to file or return data
        if destination:
            try:
                self._ensure_directory_exists(destination)
                mode = 'wb' if isinstance(export_data, bytes) else 'w'
                with open(destination, mode, encoding='utf-8' if mode == 'w' else None) as f:
                    f.write(export_data)

                return {
                    'success': True,
                    'records_processed': len(knowledge_data.get('triples', [])),
                    'file_path': destination,
                    'format': fmt,
                    'compression': compression,
                    'entity_count': len(knowledge_data.get('entities', {})),
                    'relation_count': len(knowledge_data.get('relations', {})),
                    'errors': []
                }
            except Exception as e:
                return {
                    'success': False,
                    'records_processed': 0,
                    'file_path': None,
                    'errors': [f"File write failed: {str(e)}"]
                }
        else:
            # Return data directly
            return {
                'success': True,
                'records_processed': len(knowledge_data.get('triples', [])),
                'data': export_data,
                'format': fmt,
                'compression': 'none',
                'entity_count': len(knowledge_data.get('entities', {})),
                'relation_count': len(knowledge_data.get('relations', {})),
                'errors': []
            }

    def _execute_import(self, inputs: Dict, context) -> Dict[str, Any]:
        """Import knowledge from external source."""
        source = inputs.get('source_path', self.config.get('source_path'))
        fmt = inputs.get('format', self.config.get('format', 'json'))
        validate = inputs.get('validate_on_import', self.config.get('validate_on_import', True))
        merge_strategy = inputs.get('merge_strategy', self.config.get('merge_strategy', 'merge'))

        context.update_progress(20, f"Loading data from {source}")

        # Load data from source
        try:
            data = self._load_from_source(source)
        except Exception as e:
            return {
                'success': False,
                'records_processed': 0,
                'errors': [f"Failed to load from source: {str(e)}"]
            }

        context.update_progress(40, "Detecting format and parsing")

        # Auto-detect format if not specified
        if fmt == 'json' and isinstance(data, str):
            fmt = self._detect_format(source, data)

        # Parse data based on format
        try:
            if fmt == 'json':
                knowledge_data = self._import_from_json(data)
            elif fmt in ['rdf', 'ttl']:
                knowledge_data = self._import_from_ttl(data)
            elif fmt == 'csv':
                knowledge_data = self._import_from_csv(data)
            elif fmt == 'nquads':
                knowledge_data = self._import_from_nquads(data)
            elif fmt == 'networkx':
                knowledge_data = self._import_from_networkx(data)
            else:
                raise ValueError(f"Unsupported format: {fmt}")
        except Exception as e:
            return {
                'success': False,
                'records_processed': 0,
                'errors': [f"Format parsing failed: {str(e)}"]
            }

        context.update_progress(60, "Validating data" if validate else "Skipping validation")

        # Validate if requested
        errors = []
        if validate:
            is_valid, validation_errors = self._validate_knowledge_data(knowledge_data)
            if not is_valid:
                errors.extend(validation_errors)
                if self.config.get('strict_validation', False):
                    return {
                        'success': False,
                        'records_processed': 0,
                        'errors': errors
                    }

        context.update_progress(80, f"Merging with strategy: {merge_strategy}")

        # Merge with existing knowledge
        records_processed = self._merge_knowledge(knowledge_data, merge_strategy)

        context.update_progress(100, "Import complete")

        return {
            'success': True,
            'records_processed': records_processed,
            'format': fmt,
            'merge_strategy': merge_strategy,
            'validation_errors': errors if validate else [],
            'entity_count': len(knowledge_data.get('entities', {})),
            'relation_count': len(knowledge_data.get('relations', {})),
            'errors': errors if errors else []
        }

    def _execute_transform(self, inputs: Dict, context) -> Dict[str, Any]:
        """Transform between formats without storage."""
        source = inputs.get('source_path', self.config.get('source_path'))
        destination = inputs.get('destination_path', self.config.get('destination_path'))
        source_fmt = inputs.get('source_format', self.config.get('format', 'json'))
        target_fmt = inputs.get('target_format', inputs.get('format', self.config.get('format', 'json')))
        compression = inputs.get('compression', self.config.get('compression', 'none'))

        context.update_progress(20, "Loading source data")

        # Load source data
        try:
            data = self._load_from_source(source)
        except Exception as e:
            return {
                'success': False,
                'records_processed': 0,
                'errors': [f"Failed to load from source: {str(e)}"]
            }

        context.update_progress(40, f"Parsing from {source_fmt}")

        # Parse from source format
        try:
            if source_fmt == 'json':
                knowledge_data = self._import_from_json(data)
            elif source_fmt in ['rdf', 'ttl']:
                knowledge_data = self._import_from_ttl(data)
            elif source_fmt == 'csv':
                knowledge_data = self._import_from_csv(data)
            elif source_fmt == 'nquads':
                knowledge_data = self._import_from_nquads(data)
            else:
                raise ValueError(f"Unsupported source format: {source_fmt}")
        except Exception as e:
            return {
                'success': False,
                'records_processed': 0,
                'errors': [f"Source format parsing failed: {str(e)}"]
            }

        context.update_progress(60, f"Converting to {target_fmt}")

        # Convert to target format
        try:
            if target_fmt == 'json':
                export_data = self._export_to_json(knowledge_data)
            elif target_fmt in ['rdf', 'ttl']:
                export_data = self._export_to_ttl(knowledge_data)
            elif target_fmt == 'csv':
                export_data = self._export_to_csv(knowledge_data)
            elif target_fmt == 'nquads':
                export_data = self._export_to_nquads(knowledge_data)
            else:
                raise ValueError(f"Unsupported target format: {target_fmt}")
        except Exception as e:
            return {
                'success': False,
                'records_processed': 0,
                'errors': [f"Target format conversion failed: {str(e)}"]
            }

        context.update_progress(80, "Writing to destination")

        # Apply compression and write
        if compression != 'none':
            export_data = self._apply_compression(export_data, compression)
            destination = self._update_extension_for_compression(destination, compression)

        try:
            self._ensure_directory_exists(destination)
            mode = 'wb' if isinstance(export_data, bytes) else 'w'
            with open(destination, mode, encoding='utf-8' if mode == 'w' else None) as f:
                f.write(export_data)

            return {
                'success': True,
                'records_processed': len(knowledge_data.get('triples', [])),
                'file_path': destination,
                'source_format': source_fmt,
                'target_format': target_fmt,
                'compression': compression,
                'errors': []
            }
        except Exception as e:
            return {
                'success': False,
                'records_processed': 0,
                'errors': [f"File write failed: {str(e)}"]
            }

    # =========================================================================
    # Format-specific export methods
    # =========================================================================

    def _export_to_json(self, knowledge_data: Dict, include_metadata: bool = True) -> str:
        """Export to JSON format."""
        export = {
            'entities': knowledge_data.get('entities', {}),
            'relations': knowledge_data.get('relations', {}),
            'triples': knowledge_data.get('triples', [])
        }
        if include_metadata:
            export['metadata'] = {
                'export_timestamp': datetime.utcnow().isoformat(),
                'version': self.VERSION,
                'entity_count': len(knowledge_data.get('entities', {})),
                'relation_count': len(knowledge_data.get('relations', {})),
                'triple_count': len(knowledge_data.get('triples', []))
            }
        return json.dumps(export, indent=2)

    def _export_to_ttl(self, knowledge_data: Dict, include_metadata: bool = True) -> str:
        """Export to Turtle (TTL) format."""
        lines = [
            "@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .",
            "@prefix rdfs: <http://www.w3.org/2000/0/01/rdf-schema#> .",
            "@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .",
            "@prefix kg: <http://openevolve.org/knowledge/> .",
            ""
        ]

        # Add entity definitions
        for entity_id, entity in knowledge_data.get('entities', {}).items():
            safe_id = self._safe_uri(entity_id)
            lines.append(f"kg:{safe_id} rdf:type kg:Entity ;")
            if isinstance(entity, dict) and 'name' in entity:
                lines.append(f'    rdfs:label "{self._escape_ttl(entity["name"])}" ;')
            lines.append("    .")
            lines.append("")

        # Add triples as RDF statements
        for triple in knowledge_data.get('triples', []):
            if isinstance(triple, dict):
                subj = self._safe_uri(triple.get('subject', ''))
                pred = self._safe_uri(triple.get('predicate', ''))
                obj = triple.get('object', '')
                if isinstance(obj, str) and not obj.startswith('http'):
                    obj_str = f'"{self._escape_ttl(obj)}"'
                else:
                    obj_str = self._safe_uri(obj)
                lines.append(f"kg:{subj} kg:{pred} {obj_str} .")

        if include_metadata:
            lines.append("")
            lines.append("# Metadata")
            lines.append(f'kg:export_metadata rdf:type kg:ExportMetadata ;')
            lines.append(f'    kg:export_timestamp "{datetime.utcnow().isoformat()}"^^xsd:dateTime ;')
            lines.append(f'    kg:version "{self.VERSION}" ;')
            lines.append(f'    kg:triple_count "{len(knowledge_data.get("triples", []))}"^^xsd:integer ;')
            lines.append("    .")

        return "\n".join(lines)

    def _export_to_csv(self, knowledge_data: Dict) -> str:
        """Export to CSV format (entities and relationships as separate CSVs combined)."""
        output = io.StringIO()
        writer = csv.writer(output)

        # Write entities section
        writer.writerow(['# ENTITIES'])
        writer.writerow(['id', 'name', 'type', 'properties'])
        for entity_id, entity in knowledge_data.get('entities', {}).items():
            if isinstance(entity, dict):
                writer.writerow([
                    entity_id,
                    entity.get('name', ''),
                    entity.get('type', ''),
                    json.dumps({k: v for k, v in entity.items() if k not in ['name', 'type']})
                ])
            else:
                writer.writerow([entity_id, str(entity), '', '{}'])

        writer.writerow([])

        # Write relationships section
        writer.writerow(['# RELATIONSHIPS'])
        writer.writerow(['subject', 'predicate', 'object', 'confidence', 'source'])
        for triple in knowledge_data.get('triples', []):
            if isinstance(triple, dict):
                writer.writerow([
                    triple.get('subject', ''),
                    triple.get('predicate', ''),
                    triple.get('object', ''),
                    triple.get('confidence', 1.0),
                    triple.get('source', 'unknown')
                ])
            elif isinstance(triple, tuple) and len(triple) >= 3:
                writer.writerow([triple[0], triple[1], triple[2], 1.0, 'unknown'])

        return output.getvalue()

    def _export_to_nquads(self, knowledge_data: Dict) -> str:
        """Export to N-Quads format."""
        lines = []
        graph_id = f"<http://openevolve.org/knowledge/graph/{datetime.utcnow().strftime('%Y%m%d%H%M%S')}>"

        for triple in knowledge_data.get('triples', []):
            if isinstance(triple, dict):
                subj = self._to_nquad_term(triple.get('subject', ''))
                pred = self._to_nquad_term(triple.get('predicate', ''), is_predicate=True)
                obj = self._to_nquad_term(triple.get('object', ''))
                lines.append(f"{subj} {pred} {obj} {graph_id} .")

        return "\n".join(lines)

    def _export_to_networkx(self, knowledge_data: Dict) -> Dict:
        """Export to NetworkX-compatible format."""
        nodes = []
        edges = []

        for entity_id, entity in knowledge_data.get('entities', {}).items():
            node = {'id': entity_id}
            if isinstance(entity, dict):
                node.update(entity)
            else:
                node['label'] = str(entity)
            nodes.append(node)

        for triple in knowledge_data.get('triples', []):
            if isinstance(triple, dict):
                edges.append({
                    'source': triple.get('subject'),
                    'target': triple.get('object'),
                    'relation': triple.get('predicate'),
                    'confidence': triple.get('confidence', 1.0),
                    'metadata': triple.get('metadata', {})
                })

        return {
            'directed': True,
            'multigraph': True,
            'graph': {
                'name': 'knowledge_graph',
                'exported_at': datetime.utcnow().isoformat()
            },
            'nodes': nodes,
            'edges': edges
        }

    # =========================================================================
    # Format-specific import methods
    # =========================================================================

    def _import_from_json(self, data: Union[str, Dict]) -> Dict:
        """Import from JSON format."""
        if isinstance(data, str):
            return json.loads(data)
        return data

    def _import_from_ttl(self, data: str) -> Dict:
        """Import from Turtle/TTL format (simplified parser)."""
        entities = {}
        triples = []

        lines = data.split('\n')
        current_subject = None

        for line in lines:
            line = line.strip()
            if not line or line.startswith('@') or line.startswith('#'):
                continue

            # Parse simple triples: subject predicate object .
            if line.endswith('.'):
                line = line[:-1].strip()
                parts = line.split(None, 2)
                if len(parts) >= 3:
                    subj = self._from_ttl_term(parts[0])
                    pred = self._from_ttl_term(parts[1])
                    obj = self._from_ttl_term(parts[2])
                    triples.append({
                        'subject': subj,
                        'predicate': pred,
                        'object': obj,
                        'confidence': 1.0,
                        'source': 'ttl_import'
                    })
                    entities[subj] = {'name': subj}
                    entities[obj] = {'name': obj}

        return {
            'entities': entities,
            'relations': {},
            'triples': triples
        }

    def _import_from_csv(self, data: str) -> Dict:
        """Import from CSV format."""
        entities = {}
        relations = {}
        triples = []

        reader = csv.reader(io.StringIO(data))
        section = None

        for row in reader:
            if not row:
                continue

            # Detect section headers
            if row[0].startswith('#'):
                if 'ENTITIES' in row[0]:
                    section = 'entities'
                elif 'RELATIONSHIPS' in row[0]:
                    section = 'relationships'
                continue

            # Skip header rows
            if row[0] in ['id', 'subject']:
                continue

            if section == 'entities' and len(row) >= 2:
                entity_id = row[0]
                entities[entity_id] = {
                    'name': row[1],
                    'type': row[2] if len(row) > 2 else '',
                }
                try:
                    if len(row) > 3 and row[3]:
                        entities[entity_id]['properties'] = json.loads(row[3])
                except json.JSONDecodeError:
                    pass

            elif section == 'relationships' and len(row) >= 3:
                triple = {
                    'subject': row[0],
                    'predicate': row[1],
                    'object': row[2],
                    'confidence': float(row[3]) if len(row) > 3 and row[3] else 1.0,
                    'source': row[4] if len(row) > 4 else 'csv_import'
                }
                triples.append(triple)
                relations[row[1]] = {'name': row[1]}

        return {
            'entities': entities,
            'relations': relations,
            'triples': triples
        }

    def _import_from_nquads(self, data: str) -> Dict:
        """Import from N-Quads format."""
        entities = {}
        triples = []

        for line in data.strip().split('\n'):
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            # Parse: subject predicate object graph .
            if line.endswith('.'):
                parts = line[:-1].strip().split()
                if len(parts) >= 3:
                    subj = self._from_nquad_term(parts[0])
                    pred = self._from_nquad_term(parts[1])
                    obj = self._from_nquad_term(parts[2])
                    triples.append({
                        'subject': subj,
                        'predicate': pred,
                        'object': obj,
                        'confidence': 1.0,
                        'source': 'nquads_import'
                    })
                    entities[subj] = {'name': subj}
                    entities[obj] = {'name': obj}

        return {
            'entities': entities,
            'relations': {},
            'triples': triples
        }

    def _import_from_networkx(self, data: Dict) -> Dict:
        """Import from NetworkX format."""
        entities = {}
        relations = {}
        triples = []

        for node in data.get('nodes', []):
            if isinstance(node, dict):
                entity_id = node.get('id', str(node))
                entities[entity_id] = {k: v for k, v in node.items() if k != 'id'}
            else:
                entities[str(node)] = {'name': str(node)}

        for edge in data.get('edges', []):
            if isinstance(edge, dict):
                triple = {
                    'subject': edge.get('source'),
                    'predicate': edge.get('relation', 'related_to'),
                    'object': edge.get('target'),
                    'confidence': edge.get('confidence', 1.0),
                    'metadata': edge.get('metadata', {}),
                    'source': 'networkx_import'
                }
                triples.append(triple)
                relations[triple['predicate']] = {'name': triple['predicate']}

        return {
            'entities': entities,
            'relations': relations,
            'triples': triples
        }

    # =========================================================================
    # Helper methods
    # =========================================================================

    def _get_knowledge_from_hub(self, entity_types: List[str]) -> Dict:
        """Get knowledge data from the UnifiedKGIntegrationHub."""
        if not self.hub:
            return {'entities': {}, 'relations': {}, 'triples': []}

        # Filter by entity types if specified
        entities = {}
        for entity_id, entity in self.hub.entities.items():
            if not entity_types or (isinstance(entity, dict) and entity.get('type') in entity_types):
                entities[entity_id] = entity

        # Filter triples to only include filtered entities
        entity_ids = set(entities.keys())
        triples = []
        for triple in self.hub.triples:
            triple_dict = triple.to_dict() if hasattr(triple, 'to_dict') else triple
            if triple_dict.get('subject') in entity_ids or triple_dict.get('object') in entity_ids:
                triples.append(triple_dict)

        return {
            'entities': entities,
            'relations': dict(self.hub.relations),
            'triples': triples
        }

    def _load_from_source(self, source: str) -> Union[str, bytes]:
        """Load data from file path or URL."""
        parsed = urlparse(source)

        if parsed.scheme in ('http', 'https'):
            # Load from URL
            with urllib.request.urlopen(source, timeout=30) as response:
                return response.read().decode('utf-8')
        else:
            # Load from file
            path = Path(source)
            if not path.exists():
                raise FileNotFoundError(f"Source file not found: {source}")

            # Check file size
            file_size_mb = path.stat().st_size / (1024 * 1024)
            if file_size_mb > self.MAX_FILE_SIZE_MB:
                raise ValueError(
                    f"File size ({file_size_mb:.1f} MB) exceeds maximum ({self.MAX_FILE_SIZE_MB} MB)"
                )

            # Handle compressed files
            if source.endswith('.gz'):
                with gzip.open(source, 'rt', encoding='utf-8') as f:
                    return f.read()
            elif source.endswith('.zip'):
                with zipfile.ZipFile(source, 'r') as z:
                    # Read first file in archive
                    first_file = z.namelist()[0]
                    with z.open(first_file) as f:
                        return f.read().decode('utf-8')
            else:
                with open(source, 'r', encoding='utf-8') as f:
                    return f.read()

    def _detect_format(self, source: str, data: str) -> str:
        """Auto-detect format from file extension or content."""
        ext = Path(source).suffix.lower()

        if ext in ['.json', '.jsonld']:
            return 'json'
        elif ext in ['.ttl', '.turtle']:
            return 'ttl'
        elif ext == '.csv':
            return 'csv'
        elif ext in ['.nq', '.nquads']:
            return 'nquads'
        elif ext in ['.rdf', '.xml']:
            return 'rdf'

        # Content-based detection
        if data.strip().startswith('{'):
            return 'json'
        elif '@prefix' in data or data.strip().startswith('<') and '> <' in data:
            return 'ttl'
        elif 'http' in data and '> <' in data:
            return 'nquads'

        return 'json'  # Default

    def _validate_knowledge_data(self, knowledge_data: Dict) -> tuple[bool, List[str]]:
        """Validate knowledge data structure."""
        errors = []

        if not isinstance(knowledge_data, dict):
            errors.append("Knowledge data must be a dictionary")
            return False, errors

        # Check required fields
        if 'triples' not in knowledge_data:
            errors.append("Missing required field: 'triples'")
        elif not isinstance(knowledge_data['triples'], list):
            errors.append("'triples' must be a list")

        # Validate triple structure
        for i, triple in enumerate(knowledge_data.get('triples', [])):
            if isinstance(triple, dict):
                if 'subject' not in triple:
                    errors.append(f"Triple[{i}] missing 'subject'")
                if 'predicate' not in triple:
                    errors.append(f"Triple[{i}] missing 'predicate'")
                if 'object' not in triple:
                    errors.append(f"Triple[{i}] missing 'object'")

        return len(errors) == 0, errors

    def _merge_knowledge(self, new_data: Dict, strategy: str) -> int:
        """Merge imported knowledge with existing hub data."""
        if not self.hub:
            return len(new_data.get('triples', []))

        records_processed = 0

        # Merge entities
        for entity_id, entity in new_data.get('entities', {}).items():
            if strategy == 'replace' or entity_id not in self.hub.entities:
                self.hub.entities[entity_id] = entity
                records_processed += 1
            elif strategy == 'merge':
                if entity_id in self.hub.entities:
                    if isinstance(self.hub.entities[entity_id], dict) and isinstance(entity, dict):
                        self.hub.entities[entity_id].update(entity)
                else:
                    self.hub.entities[entity_id] = entity
                records_processed += 1
            # skip_existing: do nothing if entity exists

        # Merge triples
        existing_triples = set()
        for t in self.hub.triples:
            if hasattr(t, 'subject') and hasattr(t, 'predicate') and hasattr(t, 'object'):
                existing_triples.add((t.subject, t.predicate, t.object))

        for triple_data in new_data.get('triples', []):
            if isinstance(triple_data, dict):
                triple_key = (
                    triple_data.get('subject'),
                    triple_data.get('predicate'),
                    triple_data.get('object')
                )

                if strategy == 'replace' or triple_key not in existing_triples:
                    if self.KnowledgeTriple:
                        triple = self.KnowledgeTriple.from_dict(triple_data)
                        self.hub.triples.append(triple)
                        records_processed += 1
                elif strategy == 'merge':
                    if triple_key not in existing_triples:
                        if self.KnowledgeTriple:
                            triple = self.KnowledgeTriple.from_dict(triple_data)
                            self.hub.triples.append(triple)
                            records_processed += 1
                # skip_existing: do nothing if triple exists

        return records_processed

    def _apply_compression(self, data: Union[str, bytes], compression: str) -> bytes:
        """Apply compression to data."""
        if isinstance(data, str):
            data = data.encode('utf-8')

        if compression == 'gzip':
            return gzip.compress(data)
        elif compression == 'zip':
            buffer = io.BytesIO()
            with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
                zf.writestr('knowledge_export', data)
            return buffer.getvalue()

        return data

    def _update_extension_for_compression(self, path: str, compression: str) -> str:
        """Update file extension based on compression type."""
        if compression == 'gzip' and not path.endswith('.gz'):
            return path + '.gz'
        elif compression == 'zip' and not path.endswith('.zip'):
            return path + '.zip'
        return path

    def _ensure_directory_exists(self, path: str):
        """Ensure the directory for the given path exists."""
        directory = os.path.dirname(path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)

    # =========================================================================
    # Format conversion helpers
    # =========================================================================

    def _safe_uri(self, value: str) -> str:
        """Convert value to safe URI component."""
        # Remove/replace unsafe characters
        safe = value.replace(' ', '_').replace(':', '_').replace('/', '_')
        safe = ''.join(c for c in safe if c.isalnum() or c in '_-')
        return safe[:100]  # Limit length

    def _escape_ttl(self, value: str) -> str:
        """Escape string for Turtle format."""
        return value.replace('\\', '\\\\').replace('"', '\\"').replace('\n', '\\n')

    def _to_nquad_term(self, value: str, is_predicate: bool = False) -> str:
        """Convert value to N-Quad term."""
        if value.startswith('http://') or value.startswith('https://'):
            return f"<{value}>"
        elif is_predicate:
            return f"<http://openevolve.org/knowledge/{self._safe_uri(value)}>"
        elif isinstance(value, str) and ' ' in value:
            return f'"{self._escape_ttl(value)}"'
        else:
            return f"<http://openevolve.org/knowledge/{self._safe_uri(value)}>"

    def _from_ttl_term(self, term: str) -> str:
        """Extract value from Turtle term."""
        term = term.strip()
        if term.startswith('<') and term.endswith('>'):
            uri = term[1:-1]
            # Extract local name if it's our namespace
            if 'openevolve.org/knowledge/' in uri:
                return uri.split('/')[-1].replace('_', ' ')
            return uri
        elif term.startswith('"') and term.endswith('"'):
            return term[1:-1].replace('\\"', '"').replace('\\\\', '\\')
        return term

    def _from_nquad_term(self, term: str) -> str:
        """Extract value from N-Quad term."""
        return self._from_ttl_term(term)

    # =========================================================================
    # Node interface methods
    # =========================================================================

    def get_parameter_schema(self) -> Dict[str, Any]:
        """
        Get JSON schema for node parameters.

        Returns schema for UI configuration with all parameters.
        """
        return {
            "type": "object",
            "title": "Knowledge Import/Export Configuration",
            "description": "Configure knowledge import, export, and format transformation operations",
            "properties": {
                "operation": {
                    "type": "string",
                    "title": "Operation",
                    "description": "The import/export operation to perform",
                    "enum": self.SUPPORTED_OPERATIONS,
                    "enumNames": [
                        "Export - Export knowledge to file",
                        "Import - Import knowledge from external source",
                        "Transform - Convert between formats"
                    ],
                    "default": "export"
                },
                "format": {
                    "type": "string",
                    "title": "Format",
                    "description": "Target format for the operation",
                    "enum": self.SUPPORTED_FORMATS,
                    "enumNames": [
                        "JSON - Native knowledge graph format",
                        "RDF - Resource Description Framework",
                        "TTL - Turtle (linked data format)",
                        "CSV - Tabular format",
                        "N-Quads - RDF with context",
                        "NetworkX - Python graph format"
                    ],
                    "default": "json"
                },
                "source_path": {
                    "type": "string",
                    "title": "Source Path",
                    "description": "File path or URL for import/transform operations",
                    "default": ""
                },
                "destination_path": {
                    "type": "string",
                    "title": "Destination Path",
                    "description": "Output file path for export/transform operations (omit to return data)",
                    "default": ""
                },
                "entity_types": {
                    "type": "array",
                    "title": "Entity Types",
                    "description": "Filter by entity types (empty for all)",
                    "items": {
                        "type": "string"
                    },
                    "default": []
                },
                "include_metadata": {
                    "type": "boolean",
                    "title": "Include Metadata",
                    "description": "Include export metadata (timestamps, counts, etc.)",
                    "default": True
                },
                "validate_on_import": {
                    "type": "boolean",
                    "title": "Validate on Import",
                    "description": "Validate data integrity during import",
                    "default": True
                },
                "merge_strategy": {
                    "type": "string",
                    "title": "Merge Strategy",
                    "description": "Strategy for handling existing data during import",
                    "enum": self.SUPPORTED_MERGE_STRATEGIES,
                    "enumNames": [
                        "Replace - Replace existing data",
                        "Merge - Merge with existing data",
                        "Skip Existing - Skip existing records"
                    ],
                    "default": "merge"
                },
                "compression": {
                    "type": "string",
                    "title": "Compression",
                    "description": "Compression type for export",
                    "enum": self.SUPPORTED_COMPRESSION,
                    "enumNames": [
                        "None - No compression",
                        "Gzip - Gzip compression",
                        "Zip - Zip archive"
                    ],
                    "default": "none"
                }
            }
        }

    def is_healthy(self) -> bool:
        """
        Check if the node is healthy and ready to execute.

        Returns:
            True if basic functionality is available
        """
        # Node can work with or without the hub (has fallback for transform operations)
        return True

    def get_supported_formats(self) -> List[str]:
        """Get list of supported formats."""
        return self.SUPPORTED_FORMATS.copy()

    def get_supported_operations(self) -> List[str]:
        """Get list of supported operations."""
        return self.SUPPORTED_OPERATIONS.copy()
