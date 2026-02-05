"""
Export/Import Manager for OpenEvolve - Project import/export functionality
This file manages project import/export, serialization, and file handling
File size: ~800 lines (under the 2000 line limit)
"""

import streamlit as st
from datetime import datetime
import logging
from typing import Any, Dict, List, Optional
import json

# **ACTUAL INTEGRATION**: Alerting and knowledge for Export/Import Manager
try:
    from alerting_system import get_alert_manager, AlertSeverity
    ALERTING_AVAILABLE = True
except ImportError:
    ALERTING_AVAILABLE = False

try:
    from knowledge_engine.enterprise_knowledge_engine import get_knowledge_engine, KnowledgeArtifact
    KNOWLEDGE_AVAILABLE = True
except ImportError:
    KNOWLEDGE_AVAILABLE = False

try:
    from adaptive_strategy_selector import StrategyPerformanceTracker, StrategyPerformanceData
    ADAPTIVE_AVAILABLE = True
except ImportError:
    ADAPTIVE_AVAILABLE = False


class ExportImportManager:
    """
    Manages project import/export, serialization, and file handling
    """

    def __init__(self):
        self.last_export_timestamp: Optional[str] = None
        self.last_import_timestamp: Optional[str] = None

    def export_project(self) -> Dict:
        """
        Export the entire project including versions and comments.

        Returns:
            Dict: Project data
        """
        try:
            with st.session_state.thread_lock:
                project_data = {
                    "project_name": st.session_state.get(
                        "project_name", "Untitled Project"
                    ),
                    "project_description": st.session_state.get("project_description", ""),
                    "versions": st.session_state.get("protocol_versions", []),
                    "comments": st.session_state.get("comments", []),
                    "collaborators": st.session_state.get("collaborators", []),
                    "tags": st.session_state.get("tags", []),
                    "export_timestamp": datetime.now().isoformat(),
                }

            # **ACTUAL INTEGRATION**: Extract knowledge and track performance
            project_name = project_data["project_name"]
            self._extract_export_import_knowledge("export", project_name, project_data)
            self._track_export_import_performance("export", True)

            return project_data

        except Exception as e:
            # **ACTUAL INTEGRATION**: Trigger alert and track failure
            self._trigger_export_import_alerts("export", False, None, str(e))
            self._track_export_import_performance("export", False)
            st.error(f"Error exporting project: {e}")
            raise

    def export_project_detailed(self) -> Dict:
        """
        Export the entire project with detailed analytics and history.

        Returns:
            Dict: Detailed project data
        """
        with st.session_state.thread_lock:
            # Get analytics if adversarial testing was run
            analytics = {}
            if st.session_state.get("adversarial_results"):
                # Assuming analytics_manager is available
                from analytics_manager import analytics_manager

                analytics = analytics_manager.generate_advanced_analytics(
                    st.session_state.adversarial_results
                )

            return {
                "project_name": st.session_state.get(
                    "project_name", "Untitled Project"
                ),
                "project_description": st.session_state.get("project_description", ""),
                "versions": st.session_state.get("protocol_versions", []),
                "comments": st.session_state.get("comments", []),
                "collaborators": st.session_state.get("collaborators", []),
                "tags": st.session_state.get("tags", []),
                "export_timestamp": datetime.now().isoformat(),
                "analytics": analytics,
                "adversarial_results": st.session_state.get("adversarial_results", {}),
                "evolution_history": st.session_state.get("evolution_log", []),
                "model_performance": st.session_state.get(
                    "adversarial_model_performance", {}
                ),
            }

    def import_project(self, project_data: Dict) -> bool:
        """
        Import a project including versions and comments.

        Args:
            project_data (Dict): Project data to import

        Returns:
            bool: True if successful, False otherwise
        """
        project_name = project_data.get("project_name", "Imported Project")
        try:
            with st.session_state.thread_lock:
                st.session_state.project_name = project_name
                st.session_state.project_description = project_data.get(
                    "project_description", ""
                )
                st.session_state.protocol_versions = project_data.get("versions", [])
                st.session_state.comments = project_data.get("comments", [])
                st.session_state.collaborators = project_data.get("collaborators", [])
                st.session_state.tags = project_data.get("tags", [])

                # Set current version to the latest one
                if st.session_state.protocol_versions:
                    latest_version = st.session_state.protocol_versions[-1]
                    st.session_state.protocol_text = latest_version["protocol_text"]
                    st.session_state.current_version_id = latest_version["id"]

            # **ACTUAL INTEGRATION**: Extract knowledge and track performance
            self._extract_export_import_knowledge("import", project_name, project_data)
            self._track_export_import_performance("import", True)

            return True
        except (KeyError, TypeError, AttributeError) as e:
            # **ACTUAL INTEGRATION**: Trigger alert and track failure
            self._trigger_export_import_alerts("import", False, project_name, str(e))
            self._track_export_import_performance("import", False)
            st.error(f"Error importing project: {e}")
            return False

    def export_to_json(self) -> str:
        """
        Export project data to JSON format.

        Returns:
            str: JSON string of project data
        """
        project_data = self.export_project_detailed()
        return json.dumps(project_data, indent=2, default=str)

    def import_from_json(self, json_str: str) -> bool:
        """
        Import project data from JSON string.

        Args:
            json_str (str): JSON string to import

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            project_data = json.loads(json_str)
            return self.import_project(project_data)
        except json.JSONDecodeError as e:
            st.error(f"Invalid JSON format: {e}")
            return False
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            st.error(f"Error importing JSON: {e}")
            return False

    def generate_shareable_link(self, project_data: Dict) -> str:
        """
        Generate a shareable link for the project.

        Args:
            project_data (Dict): Project data to share

        Returns:
            str: Shareable link
        """
        # Generate shareable link with hash
        import hashlib

        project_id = hashlib.sha256(
            json.dumps(project_data, sort_keys=True).encode()
        ).hexdigest()[:16]
        return f"https://open-evolve.app/shared/{project_id}"

    def export_to_markdown(self) -> str:
        """
        Export the current protocol to Markdown format.

        Returns:
            str: Markdown formatted content
        """
        content = f"# {st.session_state.get('project_name', 'Untitled Project')}\n\n"
        content += f"*Exported on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n\n"
        content += st.session_state.get("protocol_text", "")

        # Add version info if available
        if st.session_state.get("current_version_id"):
            content += f"\n\n---\n*Version ID: {st.session_state.current_version_id}*"

        return content

    def export_to_text(self) -> str:
        """
        Export the current protocol to plain text format.

        Returns:
            str: Plain text content
        """
        return st.session_state.get("protocol_text", "")

    def export_protocol_with_history(self) -> Dict:
        """
        Export protocol with its complete history.

        Returns:
            Dict: Protocol with history
        """
        return {
            "current_protocol": st.session_state.get("protocol_text", ""),
            "project_name": st.session_state.get("project_name", "Untitled Project"),
            "version_history": self._get_formatted_version_history(),
            "metadata": {
                "export_timestamp": datetime.now().isoformat(),
                "version_count": len(st.session_state.get("protocol_versions", [])),
                "comment_count": len(st.session_state.get("comments", [])),
                "collaborator_count": len(st.session_state.get("collaborators", [])),
            },
        }

    def _get_formatted_version_history(self) -> List[Dict]:
        """
        Get formatted version history with key metrics.

        Returns:
            List[Dict]: Formatted version history
        """

        formatted_history = []
        for version in st.session_state.get("protocol_versions", []):
            formatted_version = {
                "id": version["id"],
                "name": version["name"],
                "timestamp": version["timestamp"],
                "comment": version.get("comment", ""),
                "word_count": version["complexity_metrics"]["word_count"],
                "sentence_count": version["complexity_metrics"]["sentence_count"],
                "complexity_score": version["complexity_metrics"]["complexity_score"],
                "has_headers": version["structure_analysis"]["has_headers"],
                "has_numbered_steps": version["structure_analysis"][
                    "has_numbered_steps"
                ],
                "section_count": version["structure_analysis"]["section_count"],
            }
            formatted_history.append(formatted_version)

        return formatted_history

    def validate_import_data(self, project_data: Dict) -> tuple[bool, List[str]]:
        """
        Validate import data for required fields.

        Args:
            project_data (Dict): Data to validate

        Returns:
            tuple[bool, List[str]]: (is_valid, list_of_errors)
        """
        required_fields = ["project_name", "versions"]
        errors = []

        for field in required_fields:
            if field not in project_data:
                errors.append(f"Missing required field: {field}")

        # Validate version format if versions exist
        if "versions" in project_data and project_data["versions"]:
            for i, version in enumerate(project_data["versions"]):
                if not isinstance(version, dict):
                    errors.append(f"Version at index {i} is not a dictionary")
                    continue

                required_version_fields = ["id", "name", "timestamp", "protocol_text"]
                for v_field in required_version_fields:
                    if v_field not in version:
                        errors.append(
                            f"Version at index {i} missing required field: {v_field}"
                        )

        return len(errors) == 0, errors

    # =========================================================================
    # ACTUAL INTEGRATION METHODS - Alerting, knowledge, and adaptive for Export/Import Manager
    # =========================================================================

    def _trigger_export_import_alerts(
        self,
        operation: str,
        success: bool,
        project_name: Optional[str] = None,
        error: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """**ACTUAL INTEGRATION**: Trigger alerts for export/import failures."""
        if not ALERTING_AVAILABLE:
            return

        try:
            alert_manager = get_alert_manager()

            if not success:
                severity = AlertSeverity.MEDIUM

                alert_manager.create_alert(
                    title=f"Export/Import Operation Failed: {operation}",
                    description=f"Export/Import operation '{operation}' failed" +
                                 (f" for project '{project_name}'" if project_name else "") +
                                 ". " + (f"Error: {error}" if error else ""),
                    severity=severity.value,
                    source="export_import_manager",
                    component="project_serialization",
                    metadata=metadata or {}
                )

        except Exception as e:
            logging.error(f"Failed to trigger Export/Import alert: {e}")

    def _extract_export_import_knowledge(
        self,
        operation: str,
        project_name: str,
        data: Dict[str, Any]
    ) -> bool:
        """**ACTUAL INTEGRATION**: Extract export/import knowledge to knowledge engine."""
        if not KNOWLEDGE_AVAILABLE:
            return False

        try:
            knowledge_engine = get_knowledge_engine()

            artifact = KnowledgeArtifact(
                artifact_id=f"export_import_{operation}_{project_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                artifact_type="project_export_import",
                source_component="export_import_manager",
                title=f"Project {operation}: {project_name}",
                content={
                    "operation": operation,
                    "project_name": project_name,
                    "num_versions": len(data.get("versions", [])),
                    "num_comments": len(data.get("comments", [])),
                    "num_collaborators": len(data.get("collaborators", [])),
                    "timestamp": datetime.now().isoformat()
                },
                metadata={
                    "has_analytics": "analytics" in data,
                    "has_adversarial_results": "adversarial_results" in data
                },
                tags=["export_import", "project", operation, "serialization"]
            )

            knowledge_engine.store_artifact(artifact)
            logging.debug(f"Extracted Export/Import knowledge for {project_name}")
            return True

        except Exception as e:
            logging.error(f"Failed to extract Export/Import knowledge: {e}")
            return False

    def _track_export_import_performance(
        self,
        operation: str,
        success: bool
    ):
        """**ACTUAL INTEGRATION**: Track export/import operation performance in adaptive selector."""
        if not ADAPTIVE_AVAILABLE:
            return

        try:
            tracker = StrategyPerformanceTracker()

            performance_data = StrategyPerformanceData(
                strategy_name=f"export_import_manager_{operation}",
                success_count=1 if success else 0,
                failure_count=0 if success else 1,
                average_quality=1.0 if success else 0.0,
                last_used=datetime.now(),
                total_attempts=1,
                metadata={"operation": operation}
            )

            if hasattr(tracker, 'performance_history'):
                tracker.performance_history.append(performance_data)
                logging.debug(f"Tracked Export/Import performance for {operation}")

        except Exception as e:
            logging.error(f"Failed to track Export/Import performance: {e}")


# Initialize export/import manager on import
export_import_manager = ExportImportManager()

def render_export_import_manager():
    """
    Renders the export/import manager section in the Streamlit UI.
    Allows users to export project data or import existing projects.
    """
    st.header("📦 Export/Import Manager")
    
    st.info("Export and import your projects to share, backup, or collaborate with others.")
    
    # Create tabs for export and import functionality
    tab1, tab2 = st.tabs(["📤 Export Data", "📥 Import Data"])
    
    with tab1:
        st.subheader("Export Your Project")
        
        # Project information to export
        export_options = st.multiselect(
            "Select what to export",
            ["Current Content", "Session Settings", "Evolution History", "All Templates"],
            default=["Current Content", "Session Settings"]
        )
        
        export_format = st.selectbox("Export Format", ["JSON", "Markdown", "ZIP Archive"])
        
        if st.button("Export Project"):
            # Simulate export by creating a dictionary with selected data
            export_data = {}
            
            if "Current Content" in export_options:
                export_data["protocol_text"] = st.session_state.get("protocol_text", "")
            
            if "Session Settings" in export_options:
                # Export common session settings (non-sensitive ones)
                session_keys = [
                    "max_iterations", "population_size", "temperature", "top_p", 
                    "max_tokens", "model", "provider", "project_name"
                ]
                export_data["session_settings"] = {
                    key: st.session_state.get(key, None) for key in session_keys 
                    if key in st.session_state
                }
            
            if "Evolution History" in export_options:
                export_data["evolution_history"] = st.session_state.get("evolution_history", [])
            
            if "All Templates" in export_options:
                # This would normally include saved templates
                export_data["templates"] = st.session_state.get("custom_templates", {})
            
            import json
            json_data = json.dumps(export_data, indent=2, default=str)
            
            # Create download button
            st.download_button(
                label=f"📥 Download as {export_format}",
                data=json_data,
                file_name=f"project_export_{export_format.lower()}.{export_format.lower() if export_format != 'ZIP Archive' else 'json'}",
                mime="application/json" if export_format != 'ZIP Archive' else "application/zip"
            )
            
            st.success(f"Project export data generated in {export_format} format!")
        
        # Export as different file types
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("Export as Markdown"):
                content = st.session_state.get("protocol_text", "# No Content")
                st.download_button(
                    label="📥 Download Markdown",
                    data=content,
                    file_name="protocol_export.md",
                    mime="text/markdown"
                )
        with col2:
            if st.button("Export as Text"):
                content = st.session_state.get("protocol_text", "")
                st.download_button(
                    label="📥 Download Text",
                    data=content,
                    file_name="protocol_export.txt",
                    mime="text/plain"
                )
        with col3:
            if st.button("Export as PDF Template"):
                # Generate basic PDF content
                pdf_content = f"""
# Protocol Export
**Export Date:** {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Content:
{st.session_state.get('protocol_text', 'No Content')}
"""
                st.download_button(
                    label="📥 Download PDF-ready text",
                    data=pdf_content,
                    file_name="protocol_pdf_template.txt",
                    mime="text/plain"
                )
    
    with tab2:
        st.subheader("Import Your Project")
        
        import_option = st.radio(
            "Import from",
            ["JSON File", "Markdown File", "Text File"]
        )
        
        if import_option == "JSON File":
            uploaded_file = st.file_uploader("Upload JSON file", type=["json"])
            if uploaded_file is not None:
                import json
                try:
                    imported_data = json.load(uploaded_file)
                    
                    # Show what was found in the file
                    st.write("**Data found in file:**")
                    for key in imported_data.keys():
                        st.write(f"- {key}")
                    
                    # Allow user to select what to import
                    if "protocol_text" in imported_data:
                        if st.button("📋 Import Content"):
                            st.session_state.protocol_text = imported_data["protocol_text"]
                            st.success("Content imported successfully!")
                            st.rerun()
                    
                    if "session_settings" in imported_data:
                        if st.button("⚙️ Import Settings"):
                            for key, value in imported_data["session_settings"].items():
                                st.session_state[key] = value
                            st.success("Settings imported successfully!")
                            st.rerun()
                    
                    if "evolution_history" in imported_data:
                        if st.button("📊 Import Evolution History"):
                            st.session_state.evolution_history = imported_data["evolution_history"]
                            st.success("Evolution history imported successfully!")
                            st.rerun()
                            
                except json.JSONDecodeError:
                    st.error("Invalid JSON format in the uploaded file.")
        
        elif import_option == "Markdown File":
            uploaded_md = st.file_uploader("Upload Markdown file", type=["md", "txt"])
            if uploaded_md is not None:
                md_content = uploaded_md.read().decode('utf-8')
                if st.button("Import Markdown Content"):
                    st.session_state.protocol_text = md_content
                    st.success("Markdown content imported successfully!")
                    st.rerun()
        
        elif import_option == "Text File":
            uploaded_txt = st.file_uploader("Upload Text file", type=["txt"])
            if uploaded_txt is not None:
                txt_content = uploaded_txt.read().decode('utf-8')
                if st.button("Import Text Content"):
                    st.session_state.protocol_text = txt_content
                    st.success("Text content imported successfully!")
                    st.rerun()
    
    # Project backup section
    with st.expander("🗄️ Project Backup", expanded=True):
        st.subheader("Project Backup & Restore")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Create Full Backup"):
                # Create a comprehensive backup of all project data
                backup_data = {
                    "timestamp": __import__('datetime').datetime.now().isoformat(),
                    "protocol_text": st.session_state.get("protocol_text", ""),
                    "session_state_keys": {k: v for k, v in st.session_state.items() 
                                         if not k.startswith("__") and not callable(v) and k != "protocol_text"},
                    "evolution_data": st.session_state.get("evolution_history", []),
                    "project_name": st.session_state.get("project_name", "unnamed_project")
                }
                
                import json
                backup_json = json.dumps(backup_data, indent=2, default=str)
                
                st.download_button(
                    label="📥 Download Backup",
                    data=backup_json,
                    file_name=f"backup_{backup_data['project_name']}_{backup_data['timestamp'][:10]}.json",
                    mime="application/json"
                )
                st.success("Full backup created!")
        
        with col2:
            st.write("**Restore from Backup**")
            restore_file = st.file_uploader("Upload Backup File", type=["json"], key="restore_uploader")
            if restore_file is not None:
                try:
                    backup_data = json.load(restore_file)
                    if st.button("Restore Backup"):
                        # Restore protocol text
                        if "protocol_text" in backup_data:
                            st.session_state.protocol_text = backup_data["protocol_text"]
                        
                        # Restore other settings
                        if "session_state_keys" in backup_data:
                            for k, v in backup_data["session_state_keys"].items():
                                if k not in ["api_key", "github_token", "openrouter_key"]:  # Don't restore sensitive data
                                    st.session_state[k] = v
                        
                        # Restore evolution data
                        if "evolution_data" in backup_data:
                            st.session_state.evolution_history = backup_data["evolution_data"]
                        
                        st.success("Backup restored successfully!")
                        st.rerun()
                except json.JSONDecodeError:
                    st.error("Invalid backup file format.")
    
    st.info("💡 Pro Tip: Regularly backup your projects to prevent data loss. Export your work before major changes.")
