"""
Export/Import Manager for OpenEvolve - Project import/export functionality
This file manages project import/export, serialization, and file handling
File size: ~800 lines (under the 2000 line limit)
"""

import streamlit as st
from datetime import datetime
from typing import Dict, List
import json


class ExportImportManager:
    """
    Manages project import/export, serialization, and file handling
    """

    def __init__(self):
        pass

    def export_project(self) -> Dict:
        """
        Export the entire project including versions and comments.

        Returns:
            Dict: Project data
        """
        with st.session_state.thread_lock:
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
            }

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
        try:
            with st.session_state.thread_lock:
                st.session_state.project_name = project_data.get(
                    "project_name", "Imported Project"
                )
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
            return True
        except Exception as e:
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
        except Exception as e:
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
        # In a real implementation, this would generate a real shareable link
        # For now, we'll simulate it
        import hashlib

        project_id = hashlib.md5(
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


# Initialize export/import manager on import
export_import_manager = ExportImportManager()

def render_export_import_manager():
    """
    Placeholder function to render the export/import manager section in the Streamlit UI.
    This would typically allow users to export project data or import existing projects.
    """
    st.header("📦 Export/Import Manager")
    st.info("Export/Import features are under development. Stay tuned!")
    # Example of how you might use the manager:
    # if st.button("Export Project to JSON"):
    #     json_data = export_import_manager.export_to_json()
    #     st.download_button(label="Download JSON", data=json_data, file_name="project.json", mime="application/json")
    #
    # uploaded_file = st.file_uploader("Import Project from JSON", type="json")
    # if uploaded_file is not None:
    #     # Logic to read and import the file
    #     pass
