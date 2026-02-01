"""
Version Control Node for BubbleLabs Integration

Git-like version control for knowledge graphs with snapshots, branching,
merging, and rollback capabilities.

Features:
- Create snapshots/commits of knowledge graphs
- View commit history with full metadata
- Compare versions (diff functionality)
- Rollback to previous versions
- Branch and merge knowledge graphs
- Tag important versions for easy reference
- File-based fallback when version control system unavailable
"""

from typing import Dict, Any, List, Optional
from datetime import datetime
import json
import os
import hashlib
from pathlib import Path
from .base_node import BubbleLabsNode, NodeExecutionError


class VersionControlNode(BubbleLabsNode):
    """
    Git-like version control for knowledge graphs with snapshots and rollback.

    Supports:
    - commit: Create snapshots/commits of knowledge graphs
    - history: View commit history
    - diff: Compare versions
    - rollback: Rollback to previous versions
    - branch: Create branches
    - merge: Merge branches
    - tag: Tag important versions
    - checkout: Checkout specific versions

    Includes file-based fallback when KnowledgeVersionControl is not available.
    """

    # Node metadata
    DISPLAY_NAME = "Version Control"
    DESCRIPTION = "Git-like version control for knowledge graphs with snapshots and rollback"
    ICON = "version-control"
    CATEGORY = "management"
    VERSION = "1.0.0"

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)

        # Safe import of KnowledgeVersionControl
        self.KnowledgeVersionControl = None
        self.UnifiedKGIntegrationHub = None
        self.version_control = None
        self.hub = None

        vc_module = self.safe_import(
            'knowledge_engine.version_control',
            fallback_value=None,
            error_msg="KnowledgeVersionControl not available, using file-based fallback"
        )

        if vc_module:
            self.KnowledgeVersionControl = getattr(vc_module, 'KnowledgeVersionControl', None)

        hub_module = self.safe_import(
            'knowledge_engine.unified_kg_integration_hub',
            fallback_value=None,
            error_msg="UnifiedKGIntegrationHub not available"
        )

        if hub_module:
            self.UnifiedKGIntegrationHub = getattr(hub_module, 'UnifiedKGIntegrationHub', None)

        # Initialize version control if available
        if self.KnowledgeVersionControl:
            try:
                self.version_control = self.KnowledgeVersionControl()
                self.logger.info("KnowledgeVersionControl initialized successfully")
            except Exception as e:
                self.logger.warning(f"Could not initialize KnowledgeVersionControl: {e}")
                self.version_control = None

        # Initialize hub if available
        if self.UnifiedKGIntegrationHub:
            try:
                self.hub = self.UnifiedKGIntegrationHub()
                self.logger.info("UnifiedKGIntegrationHub initialized successfully")
            except Exception as e:
                self.logger.warning(f"Could not initialize UnifiedKGIntegrationHub: {e}")
                self.hub = None

        # Set up fallback storage path
        self.fallback_storage_path = self.config.get(
            'fallback_storage_path',
            'version_control_fallback'
        )
        self._ensure_fallback_storage()

        # Valid operations
        self.valid_operations = [
            'commit', 'history', 'diff', 'rollback',
            'branch', 'merge', 'tag', 'checkout'
        ]

    def _ensure_fallback_storage(self):
        """Ensure fallback storage directory exists."""
        try:
            Path(self.fallback_storage_path).mkdir(parents=True, exist_ok=True)
        except Exception as e:
            self.logger.warning(f"Could not create fallback storage: {e}")

    def _get_kg_path(self, kg_id: str) -> Path:
        """Get path for knowledge graph storage."""
        return Path(self.fallback_storage_path) / kg_id

    def _get_versions_path(self, kg_id: str) -> Path:
        """Get path for versions file."""
        return self._get_kg_path(kg_id) / 'versions.json'

    def _get_branches_path(self, kg_id: str) -> Path:
        """Get path for branches file."""
        return self._get_kg_path(kg_id) / 'branches.json'

    def _get_tags_path(self, kg_id: str) -> Path:
        """Get path for tags file."""
        return self._get_kg_path(kg_id) / 'tags.json'

    def _generate_version_id(self, kg_id: str, timestamp: str) -> str:
        """Generate a unique version ID."""
        content = f"{kg_id}:{timestamp}:{datetime.now().microsecond}"
        return f"v{hashlib.sha256(content.encode()).hexdigest()[:12]}"

    def _load_fallback_data(self, kg_id: str) -> Dict[str, Any]:
        """Load fallback version control data for a knowledge graph."""
        kg_path = self._get_kg_path(kg_id)
        versions_path = self._get_versions_path(kg_id)
        branches_path = self._get_branches_path(kg_id)
        tags_path = self._get_tags_path(kg_id)

        # Ensure directory exists
        kg_path.mkdir(parents=True, exist_ok=True)

        data = {
            'versions': {},
            'branches': {'main': {'current_version': None, 'created_at': datetime.now().isoformat()}},
            'tags': {},
            'current_branch': 'main'
        }

        try:
            if versions_path.exists():
                with open(versions_path, 'r') as f:
                    data['versions'] = json.load(f)
            if branches_path.exists():
                with open(branches_path, 'r') as f:
                    data['branches'] = json.load(f)
            if tags_path.exists():
                with open(tags_path, 'r') as f:
                    data['tags'] = json.load(f)
        except Exception as e:
            self.logger.warning(f"Could not load fallback data: {e}")

        return data

    def _save_fallback_data(self, kg_id: str, data: Dict[str, Any]):
        """Save fallback version control data for a knowledge graph."""
        versions_path = self._get_versions_path(kg_id)
        branches_path = self._get_branches_path(kg_id)
        tags_path = self._get_tags_path(kg_id)

        try:
            with open(versions_path, 'w') as f:
                json.dump(data.get('versions', {}), f, indent=2)
            with open(branches_path, 'w') as f:
                json.dump(data.get('branches', {}), f, indent=2)
            with open(tags_path, 'w') as f:
                json.dump(data.get('tags', {}), f, indent=2)
        except Exception as e:
            self.logger.error(f"Could not save fallback data: {e}")
            raise NodeExecutionError(
                node_name=self.get_display_name(),
                message=f"Failed to save version control data: {e}",
                details={'knowledge_graph_id': kg_id}
            ) from e

    def validate_inputs(self, inputs: Dict) -> List[str]:
        """
        Validate input parameters.

        Required:
            - operation: str - One of [commit, history, diff, rollback, branch, merge, tag, checkout]
            - knowledge_graph_id: str - The knowledge graph to operate on

        Optional (operation-specific):
            - commit_message: str - For commit operations
            - version_id: str - For checkout/rollback operations
            - branch_name: str - For branch operations
            - tag_name: str - For tag operations
            - compare_from: str - Base version for diff
            - compare_to: str - Target version for diff
            - include_metadata: bool - Include full metadata in output
        """
        errors = []

        # Check required fields
        if 'operation' not in inputs:
            errors.append("Missing required field: 'operation'")
        elif inputs['operation'] not in self.valid_operations:
            errors.append(
                f"Invalid operation: '{inputs['operation']}'. "
                f"Must be one of: {', '.join(self.valid_operations)}"
            )

        if 'knowledge_graph_id' not in inputs:
            errors.append("Missing required field: 'knowledge_graph_id'")
        elif not isinstance(inputs['knowledge_graph_id'], str):
            errors.append("'knowledge_graph_id' must be a string")
        elif len(inputs['knowledge_graph_id'].strip()) == 0:
            errors.append("'knowledge_graph_id' cannot be empty")

        # Operation-specific validation
        operation = inputs.get('operation', '')

        if operation == 'commit' and 'commit_message' not in inputs:
            errors.append("'commit_message' required for commit operation")

        if operation in ['checkout', 'rollback'] and 'version_id' not in inputs:
            errors.append("'version_id' required for checkout/rollback operation")

        if operation == 'branch' and 'branch_name' not in inputs:
            errors.append("'branch_name' required for branch operation")

        if operation == 'tag' and 'tag_name' not in inputs:
            errors.append("'tag_name' required for tag operation")

        if operation == 'diff':
            if 'compare_from' not in inputs:
                errors.append("'compare_from' required for diff operation")
            if 'compare_to' not in inputs:
                errors.append("'compare_to' required for diff operation")

        # Validate boolean fields
        for bool_field in ['include_metadata']:
            if bool_field in inputs:
                if not isinstance(inputs[bool_field], bool):
                    errors.append(f"'{bool_field}' must be a boolean")

        return errors

    def execute(self, inputs: Dict, context) -> Dict[str, Any]:
        """
        Execute version control operations on knowledge graphs.

        Args:
            inputs: Operation parameters (operation, knowledge_graph_id, etc.)
            context: Workflow state for tracking progress and artifacts

        Returns:
            Dict containing operation results (version_id, timestamp, changes, etc.)

        Raises:
            NodeExecutionError: If operation fails
        """
        operation = inputs['operation']
        kg_id = inputs['knowledge_graph_id']
        include_metadata = inputs.get(
            'include_metadata',
            self.config.get('include_metadata', True)
        )

        context.update_progress(10, f"Starting {operation} operation on {kg_id}")
        self.logger.info(f"Executing version control operation: {operation} on {kg_id}")

        try:
            # Dispatch to appropriate handler
            if operation == 'commit':
                result = self._execute_commit(inputs, context)
            elif operation == 'history':
                result = self._execute_history(inputs, context)
            elif operation == 'diff':
                result = self._execute_diff(inputs, context)
            elif operation == 'rollback':
                result = self._execute_rollback(inputs, context)
            elif operation == 'branch':
                result = self._execute_branch(inputs, context)
            elif operation == 'merge':
                result = self._execute_merge(inputs, context)
            elif operation == 'tag':
                result = self._execute_tag(inputs, context)
            elif operation == 'checkout':
                result = self._execute_checkout(inputs, context)
            else:
                raise NodeExecutionError(
                    node_name=self.get_display_name(),
                    message=f"Unknown operation: {operation}",
                    details={'operation': operation}
                )

            # Add metadata if requested
            if include_metadata:
                result['metadata'] = {
                    'operation': operation,
                    'knowledge_graph_id': kg_id,
                    'timestamp': datetime.now().isoformat(),
                    'version_control_available': self.version_control is not None,
                    'hub_available': self.hub is not None
                }

            # Store artifacts in context
            context.add_artifact('version_control', {
                'operation': operation,
                'knowledge_graph_id': kg_id,
                'version_id': result.get('version_id'),
                'success': result.get('success', True)
            })

            context.update_progress(100, f"{operation} operation completed successfully")

            self.logger.info(f"Version control {operation} completed successfully")

            return result

        except NodeExecutionError:
            raise
        except Exception as e:
            self.logger.error(f"Version control {operation} failed: {e}", exc_info=True)
            raise NodeExecutionError(
                node_name=self.get_display_name(),
                message=f"Version control operation failed: {e}",
                details={
                    'operation': operation,
                    'knowledge_graph_id': kg_id,
                    'exception_type': type(e).__name__
                }
            ) from e

    def _execute_commit(self, inputs: Dict, context) -> Dict[str, Any]:
        """Create a new commit/snapshot of the knowledge graph."""
        kg_id = inputs['knowledge_graph_id']
        commit_message = inputs['commit_message']
        author = inputs.get('author', 'system')

        context.update_progress(30, "Creating commit")

        if self.version_control:
            # Use primary version control system
            try:
                version = self.version_control.commit(
                    kg_id=kg_id,
                    message=commit_message,
                    author=author
                )
                return {
                    'version_id': version.id,
                    'timestamp': version.timestamp.isoformat(),
                    'message': version.message,
                    'author': version.author,
                    'changes': version.changes,
                    'success': True
                }
            except Exception as e:
                self.logger.warning(f"Primary version control failed, using fallback: {e}")

        # Fallback implementation
        context.update_progress(50, "Using file-based fallback for commit")

        data = self._load_fallback_data(kg_id)
        timestamp = datetime.now().isoformat()
        version_id = self._generate_version_id(kg_id, timestamp)

        # Get current knowledge graph from hub if available
        kg_data = {}
        if self.hub:
            try:
                kg_data = self.hub.get_knowledge_graph(kg_id)
            except Exception as e:
                self.logger.warning(f"Could not retrieve KG from hub: {e}")

        # Create version entry
        version = {
            'id': version_id,
            'timestamp': timestamp,
            'message': commit_message,
            'author': author,
            'knowledge_graph_id': kg_id,
            'knowledge_graph_data': kg_data,
            'changes': self._detect_changes(kg_id, data, kg_data)
        }

        data['versions'][version_id] = version

        # Update current branch pointer
        current_branch = data.get('current_branch', 'main')
        data['branches'][current_branch]['current_version'] = version_id

        self._save_fallback_data(kg_id, data)

        context.update_progress(80, f"Commit {version_id} created")

        return {
            'version_id': version_id,
            'timestamp': timestamp,
            'message': commit_message,
            'author': author,
            'changes': version['changes'],
            'success': True,
            'using_fallback': True
        }

    def _execute_history(self, inputs: Dict, context) -> Dict[str, Any]:
        """Get commit history for the knowledge graph."""
        kg_id = inputs['knowledge_graph_id']
        limit = inputs.get('limit', 50)

        context.update_progress(30, "Retrieving commit history")

        if self.version_control:
            try:
                history = self.version_control.get_history(kg_id, limit=limit)
                return {
                    'commits': [
                        {
                            'version_id': v.id,
                            'timestamp': v.timestamp.isoformat(),
                            'message': v.message,
                            'author': v.author,
                            'changes_count': len(v.changes)
                        }
                        for v in history
                    ],
                    'total_commits': len(history),
                    'success': True
                }
            except Exception as e:
                self.logger.warning(f"Primary version control failed, using fallback: {e}")

        # Fallback implementation
        context.update_progress(50, "Using file-based fallback for history")

        data = self._load_fallback_data(kg_id)
        versions = data.get('versions', {})

        commits = sorted(
            [
                {
                    'version_id': vid,
                    'timestamp': v['timestamp'],
                    'message': v['message'],
                    'author': v.get('author', 'system'),
                    'changes_count': len(v.get('changes', []))
                }
                for vid, v in versions.items()
            ],
            key=lambda x: x['timestamp'],
            reverse=True
        )[:limit]

        context.update_progress(80, f"Retrieved {len(commits)} commits")

        return {
            'commits': commits,
            'total_commits': len(commits),
            'success': True,
            'using_fallback': True
        }

    def _execute_diff(self, inputs: Dict, context) -> Dict[str, Any]:
        """Compare two versions of the knowledge graph."""
        kg_id = inputs['knowledge_graph_id']
        compare_from = inputs['compare_from']
        compare_to = inputs['compare_to']

        context.update_progress(30, f"Computing diff from {compare_from} to {compare_to}")

        if self.version_control:
            try:
                diff = self.version_control.diff(kg_id, compare_from, compare_to)
                return {
                    'compare_from': compare_from,
                    'compare_to': compare_to,
                    'added': diff.added,
                    'removed': diff.removed,
                    'modified': diff.modified,
                    'statistics': diff.statistics,
                    'success': True
                }
            except Exception as e:
                self.logger.warning(f"Primary version control failed, using fallback: {e}")

        # Fallback implementation
        context.update_progress(50, "Using file-based fallback for diff")

        data = self._load_fallback_data(kg_id)
        versions = data.get('versions', {})

        from_version = versions.get(compare_from, {})
        to_version = versions.get(compare_to, {})

        from_kg = from_version.get('knowledge_graph_data', {})
        to_kg = to_version.get('knowledge_graph_data', {})

        diff_result = self._compute_diff(from_kg, to_kg)

        context.update_progress(80, "Diff computation complete")

        return {
            'compare_from': compare_from,
            'compare_to': compare_to,
            'added': diff_result['added'],
            'removed': diff_result['removed'],
            'modified': diff_result['modified'],
            'statistics': diff_result['statistics'],
            'success': True,
            'using_fallback': True
        }

    def _execute_rollback(self, inputs: Dict, context) -> Dict[str, Any]:
        """Rollback to a previous version."""
        kg_id = inputs['knowledge_graph_id']
        version_id = inputs['version_id']

        context.update_progress(30, f"Rolling back to version {version_id}")

        if self.version_control:
            try:
                result = self.version_control.rollback(kg_id, version_id)
                return {
                    'version_id': version_id,
                    'previous_version': result.previous_version,
                    'timestamp': datetime.now().isoformat(),
                    'restored_entities': result.restored_entities,
                    'success': True
                }
            except Exception as e:
                self.logger.warning(f"Primary version control failed, using fallback: {e}")

        # Fallback implementation
        context.update_progress(50, "Using file-based fallback for rollback")

        data = self._load_fallback_data(kg_id)
        versions = data.get('versions', {})

        if version_id not in versions:
            raise NodeExecutionError(
                node_name=self.get_display_name(),
                message=f"Version {version_id} not found",
                details={'version_id': version_id, 'knowledge_graph_id': kg_id}
            )

        version = versions[version_id]
        kg_data = version.get('knowledge_graph_data', {})

        # Restore to hub if available
        if self.hub:
            try:
                self.hub.update_knowledge_graph(kg_id, kg_data)
            except Exception as e:
                self.logger.warning(f"Could not restore to hub: {e}")

        # Update branch pointer
        current_branch = data.get('current_branch', 'main')
        previous_version = data['branches'][current_branch].get('current_version')
        data['branches'][current_branch]['current_version'] = version_id

        self._save_fallback_data(kg_id, data)

        context.update_progress(80, f"Rolled back to version {version_id}")

        return {
            'version_id': version_id,
            'previous_version': previous_version,
            'timestamp': datetime.now().isoformat(),
            'restored_entities': len(kg_data.get('entities', [])),
            'success': True,
            'using_fallback': True
        }

    def _execute_branch(self, inputs: Dict, context) -> Dict[str, Any]:
        """Create a new branch."""
        kg_id = inputs['knowledge_graph_id']
        branch_name = inputs['branch_name']
        from_version = inputs.get('from_version')

        context.update_progress(30, f"Creating branch {branch_name}")

        if self.version_control:
            try:
                branch = self.version_control.create_branch(
                    kg_id=kg_id,
                    branch_name=branch_name,
                    from_version=from_version
                )
                return {
                    'branch_name': branch.name,
                    'created_from': branch.created_from,
                    'timestamp': branch.created_at.isoformat(),
                    'success': True
                }
            except Exception as e:
                self.logger.warning(f"Primary version control failed, using fallback: {e}")

        # Fallback implementation
        context.update_progress(50, "Using file-based fallback for branch")

        data = self._load_fallback_data(kg_id)

        if branch_name in data['branches']:
            raise NodeExecutionError(
                node_name=self.get_display_name(),
                message=f"Branch {branch_name} already exists",
                details={'branch_name': branch_name, 'knowledge_graph_id': kg_id}
            )

        current_branch = data.get('current_branch', 'main')
        current_version = data['branches'][current_branch].get('current_version')

        data['branches'][branch_name] = {
            'current_version': from_version or current_version,
            'created_at': datetime.now().isoformat(),
            'created_from': from_version or current_version,
            'parent_branch': current_branch
        }

        self._save_fallback_data(kg_id, data)

        context.update_progress(80, f"Branch {branch_name} created")

        return {
            'branch_name': branch_name,
            'created_from': from_version or current_version,
            'timestamp': datetime.now().isoformat(),
            'success': True,
            'using_fallback': True
        }

    def _execute_merge(self, inputs: Dict, context) -> Dict[str, Any]:
        """Merge one branch into another."""
        kg_id = inputs['knowledge_graph_id']
        source_branch = inputs.get('source_branch', 'main')
        target_branch = inputs.get('target_branch')

        if not target_branch:
            raise NodeExecutionError(
                node_name=self.get_display_name(),
                message="'target_branch' required for merge operation",
                details={'operation': 'merge'}
            )

        context.update_progress(30, f"Merging {source_branch} into {target_branch}")

        if self.version_control:
            try:
                merge_result = self.version_control.merge(
                    kg_id=kg_id,
                    source_branch=source_branch,
                    target_branch=target_branch
                )
                return {
                    'source_branch': source_branch,
                    'target_branch': target_branch,
                    'merged_version': merge_result.merged_version,
                    'conflicts': merge_result.conflicts,
                    'timestamp': datetime.now().isoformat(),
                    'success': True
                }
            except Exception as e:
                self.logger.warning(f"Primary version control failed, using fallback: {e}")

        # Fallback implementation
        context.update_progress(50, "Using file-based fallback for merge")

        data = self._load_fallback_data(kg_id)

        if source_branch not in data['branches']:
            raise NodeExecutionError(
                node_name=self.get_display_name(),
                message=f"Source branch {source_branch} not found",
                details={'source_branch': source_branch}
            )

        if target_branch not in data['branches']:
            raise NodeExecutionError(
                node_name=self.get_display_name(),
                message=f"Target branch {target_branch} not found",
                details={'target_branch': target_branch}
            )

        # Create merge commit
        source_version = data['branches'][source_branch].get('current_version')
        target_version = data['branches'][target_branch].get('current_version')

        timestamp = datetime.now().isoformat()
        merge_version_id = self._generate_version_id(kg_id, timestamp)

        merge_commit = {
            'id': merge_version_id,
            'timestamp': timestamp,
            'message': f"Merge {source_branch} into {target_branch}",
            'author': 'system',
            'knowledge_graph_id': kg_id,
            'parents': [source_version, target_version],
            'merge_info': {
                'source_branch': source_branch,
                'target_branch': target_branch
            }
        }

        data['versions'][merge_version_id] = merge_commit
        data['branches'][target_branch]['current_version'] = merge_version_id

        self._save_fallback_data(kg_id, data)

        context.update_progress(80, f"Merged {source_branch} into {target_branch}")

        return {
            'source_branch': source_branch,
            'target_branch': target_branch,
            'merged_version': merge_version_id,
            'conflicts': [],  # Simplified - no conflict detection in fallback
            'timestamp': timestamp,
            'success': True,
            'using_fallback': True
        }

    def _execute_tag(self, inputs: Dict, context) -> Dict[str, Any]:
        """Create a tag for a version."""
        kg_id = inputs['knowledge_graph_id']
        tag_name = inputs['tag_name']
        version_id = inputs.get('version_id')
        tag_message = inputs.get('tag_message', '')

        context.update_progress(30, f"Creating tag {tag_name}")

        if self.version_control:
            try:
                tag = self.version_control.create_tag(
                    kg_id=kg_id,
                    tag_name=tag_name,
                    version_id=version_id,
                    message=tag_message
                )
                return {
                    'tag_name': tag.name,
                    'version_id': tag.version_id,
                    'message': tag.message,
                    'timestamp': tag.created_at.isoformat(),
                    'success': True
                }
            except Exception as e:
                self.logger.warning(f"Primary version control failed, using fallback: {e}")

        # Fallback implementation
        context.update_progress(50, "Using file-based fallback for tag")

        data = self._load_fallback_data(kg_id)

        # Determine which version to tag
        if version_id:
            if version_id not in data['versions']:
                raise NodeExecutionError(
                    node_name=self.get_display_name(),
                    message=f"Version {version_id} not found",
                    details={'version_id': version_id}
                )
        else:
            current_branch = data.get('current_branch', 'main')
            version_id = data['branches'][current_branch].get('current_version')

            if not version_id:
                raise NodeExecutionError(
                    node_name=self.get_display_name(),
                    message="No current version to tag",
                    details={'knowledge_graph_id': kg_id}
                )

        data['tags'][tag_name] = {
            'version_id': version_id,
            'message': tag_message,
            'created_at': datetime.now().isoformat()
        }

        self._save_fallback_data(kg_id, data)

        context.update_progress(80, f"Tag {tag_name} created")

        return {
            'tag_name': tag_name,
            'version_id': version_id,
            'message': tag_message,
            'timestamp': datetime.now().isoformat(),
            'success': True,
            'using_fallback': True
        }

    def _execute_checkout(self, inputs: Dict, context) -> Dict[str, Any]:
        """Checkout a specific version or branch."""
        kg_id = inputs['knowledge_graph_id']
        version_id = inputs.get('version_id')
        branch_name = inputs.get('branch_name')

        if not version_id and not branch_name:
            raise NodeExecutionError(
                node_name=self.get_display_name(),
                message="Either 'version_id' or 'branch_name' required for checkout",
                details={'operation': 'checkout'}
            )

        context.update_progress(30, f"Checking out {version_id or branch_name}")

        if self.version_control:
            try:
                result = self.version_control.checkout(
                    kg_id=kg_id,
                    version_id=version_id,
                    branch_name=branch_name
                )
                return {
                    'version_id': result.version_id,
                    'branch': result.branch,
                    'timestamp': datetime.now().isoformat(),
                    'success': True
                }
            except Exception as e:
                self.logger.warning(f"Primary version control failed, using fallback: {e}")

        # Fallback implementation
        context.update_progress(50, "Using file-based fallback for checkout")

        data = self._load_fallback_data(kg_id)

        if branch_name:
            if branch_name not in data['branches']:
                raise NodeExecutionError(
                    node_name=self.get_display_name(),
                    message=f"Branch {branch_name} not found",
                    details={'branch_name': branch_name}
                )
            data['current_branch'] = branch_name
            version_id = data['branches'][branch_name].get('current_version')
        elif version_id:
            if version_id not in data['versions']:
                raise NodeExecutionError(
                    node_name=self.get_display_name(),
                    message=f"Version {version_id} not found",
                    details={'version_id': version_id}
                )
            # Update current branch pointer
            current_branch = data.get('current_branch', 'main')
            data['branches'][current_branch]['current_version'] = version_id

        self._save_fallback_data(kg_id, data)

        context.update_progress(80, f"Checked out {version_id or branch_name}")

        return {
            'version_id': version_id,
            'branch': data.get('current_branch', 'main'),
            'timestamp': datetime.now().isoformat(),
            'success': True,
            'using_fallback': True
        }

    def _detect_changes(self, kg_id: str, data: Dict, current_kg: Dict) -> List[Dict]:
        """Detect changes between versions (simplified)."""
        changes = []

        # Get previous version
        current_branch = data.get('current_branch', 'main')
        prev_version_id = data['branches'][current_branch].get('current_version')

        if prev_version_id and prev_version_id in data['versions']:
            prev_kg = data['versions'][prev_version_id].get('knowledge_graph_data', {})
            diff = self._compute_diff(prev_kg, current_kg)

            for item in diff['added']:
                changes.append({'type': 'added', 'item': item})
            for item in diff['removed']:
                changes.append({'type': 'removed', 'item': item})
            for item in diff['modified']:
                changes.append({'type': 'modified', 'item': item})
        else:
            # First commit - everything is added
            changes.append({'type': 'initial_commit', 'entities_count': len(current_kg.get('entities', []))})

        return changes

    def _compute_diff(self, from_kg: Dict, to_kg: Dict) -> Dict[str, Any]:
        """Compute diff between two knowledge graphs."""
        from_entities = {e.get('id', str(i)): e for i, e in enumerate(from_kg.get('entities', []))}
        to_entities = {e.get('id', str(i)): e for i, e in enumerate(to_kg.get('entities', []))}

        from_relations = {
            f"{r.get('source', '')}-{r.get('target', '')}-{r.get('type', '')}": r
            for r in from_kg.get('relations', [])
        }
        to_relations = {
            f"{r.get('source', '')}-{r.get('target', '')}-{r.get('type', '')}": r
            for r in to_kg.get('relations', [])
        }

        added_entities = [e for eid, e in to_entities.items() if eid not in from_entities]
        removed_entities = [e for eid, e in from_entities.items() if eid not in to_entities]
        modified_entities = []

        for eid in from_entities:
            if eid in to_entities:
                if json.dumps(from_entities[eid], sort_keys=True) != json.dumps(to_entities[eid], sort_keys=True):
                    modified_entities.append({
                        'id': eid,
                        'from': from_entities[eid],
                        'to': to_entities[eid]
                    })

        added_relations = [r for rid, r in to_relations.items() if rid not in from_relations]
        removed_relations = [r for rid, r in from_relations.items() if rid not in to_relations]

        return {
            'added': {
                'entities': added_entities,
                'relations': added_relations
            },
            'removed': {
                'entities': removed_entities,
                'relations': removed_relations
            },
            'modified': {
                'entities': modified_entities
            },
            'statistics': {
                'entities_added': len(added_entities),
                'entities_removed': len(removed_entities),
                'entities_modified': len(modified_entities),
                'relations_added': len(added_relations),
                'relations_removed': len(removed_relations)
            }
        }

    def get_parameter_schema(self) -> Dict[str, Any]:
        """
        Get JSON schema for node parameters.

        Returns JSON schema for UI configuration panel.
        """
        return {
            "type": "object",
            "title": "Version Control Configuration",
            "description": "Configure Git-like version control for knowledge graphs",
            "properties": {
                "operation": {
                    "type": "string",
                    "title": "Operation",
                    "description": "Version control operation to perform",
                    "enum": ["commit", "history", "diff", "rollback", "branch", "merge", "tag", "checkout"],
                    "enumNames": [
                        "Commit - Create a snapshot",
                        "History - View commit history",
                        "Diff - Compare versions",
                        "Rollback - Restore previous version",
                        "Branch - Create a branch",
                        "Merge - Merge branches",
                        "Tag - Tag a version",
                        "Checkout - Switch to version/branch"
                    ],
                    "default": "commit"
                },
                "knowledge_graph_id": {
                    "type": "string",
                    "title": "Knowledge Graph ID",
                    "description": "The knowledge graph to version control",
                    "default": ""
                },
                "commit_message": {
                    "type": "string",
                    "title": "Commit Message",
                    "description": "Message describing the commit (for commit operation)",
                    "default": ""
                },
                "version_id": {
                    "type": "string",
                    "title": "Version ID",
                    "description": "Version to checkout or rollback to",
                    "default": ""
                },
                "branch_name": {
                    "type": "string",
                    "title": "Branch Name",
                    "description": "Branch name for branch operations",
                    "default": ""
                },
                "tag_name": {
                    "type": "string",
                    "title": "Tag Name",
                    "description": "Tag name for tagging operations",
                    "default": ""
                },
                "compare_from": {
                    "type": "string",
                    "title": "Compare From",
                    "description": "Base version for diff operation",
                    "default": ""
                },
                "compare_to": {
                    "type": "string",
                    "title": "Compare To",
                    "description": "Target version for diff operation",
                    "default": ""
                },
                "include_metadata": {
                    "type": "boolean",
                    "title": "Include Metadata",
                    "description": "Include full metadata in output",
                    "default": True
                },
                "fallback_storage_path": {
                    "type": "string",
                    "title": "Fallback Storage Path",
                    "description": "Path for file-based fallback storage",
                    "default": "version_control_fallback"
                }
            },
            "required": ["operation", "knowledge_graph_id"]
        }

    def is_healthy(self) -> bool:
        """
        Check if the node is healthy and ready to execute.

        Returns:
            True if node is healthy (has fallback capability)
        """
        try:
            # Node can work with or without version control (has fallback)
            self._ensure_fallback_storage()
            return True
        except Exception:
            return False

    def get_available_operations(self) -> List[str]:
        """
        Get list of available version control operations.

        Returns:
            List of operation names
        """
        return self.valid_operations.copy()

    def get_version_control_status(self, kg_id: str) -> Dict[str, Any]:
        """
        Get version control status for a knowledge graph.

        Args:
            kg_id: Knowledge graph identifier

        Returns:
            Status information including current branch, version count, etc.
        """
        try:
            data = self._load_fallback_data(kg_id)

            return {
                'knowledge_graph_id': kg_id,
                'current_branch': data.get('current_branch', 'main'),
                'total_versions': len(data.get('versions', {})),
                'total_branches': len(data.get('branches', {})),
                'total_tags': len(data.get('tags', {})),
                'branches': list(data.get('branches', {}).keys()),
                'tags': list(data.get('tags', {}).keys()),
                'version_control_available': self.version_control is not None
            }
        except Exception as e:
            self.logger.error(f"Failed to get version control status: {e}")
            return {
                'knowledge_graph_id': kg_id,
                'error': str(e),
                'version_control_available': False
            }
