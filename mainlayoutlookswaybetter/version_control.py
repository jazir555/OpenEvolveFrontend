"""
Version Control for OpenEvolve - Version history and management
This file manages protocol versions, branching, and version-related features
File size: ~1000 lines (under the 2000 line limit)
"""

import streamlit as st
import uuid
from datetime import datetime
from typing import Dict, List, Optional
from session_utils import calculate_protocol_complexity, extract_protocol_structure


class VersionControl:
    """
    Manages protocol versions, history, branching, and version-related utilities
    """

    def __init__(self):
        # Ensure protocol_versions exists in session state
        if "protocol_versions" not in st.session_state:
            st.session_state.protocol_versions = []

    def create_new_version(
        self, protocol_text: str, version_name: str = "", comment: str = ""
    ) -> str:
        """
        Create a new version of the protocol.

        Args:
            protocol_text (str): The protocol text to save
            version_name (str): Optional name for the version
            comment (str): Optional comment about the changes

        Returns:
            str: Version ID of the created version
        """
        version_id = str(uuid.uuid4())
        timestamp = datetime.now().isoformat()

        version = {
            "id": version_id,
            "name": version_name
            or f"Version {len(st.session_state.protocol_versions) + 1}",
            "timestamp": timestamp,
            "protocol_text": protocol_text,
            "comment": comment,
            "author": "Current User",  # In a real implementation, this would be the actual user
            "complexity_metrics": calculate_protocol_complexity(protocol_text),
            "structure_analysis": extract_protocol_structure(protocol_text),
        }

        with st.session_state.thread_lock:
            st.session_state.protocol_versions.append(version)
            st.session_state.current_version_id = version_id

        return version_id

    def load_version(self, version_id: str) -> bool:
        """
        Load a specific version of the protocol.

        Args:
            version_id (str): ID of the version to load

        Returns:
            bool: True if successful, False otherwise
        """
        with st.session_state.thread_lock:
            for version in st.session_state.protocol_versions:
                if version["id"] == version_id:
                    st.session_state.protocol_text = version["protocol_text"]
                    st.session_state.current_version_id = version_id
                    return True
        return False

    def get_version_history(self) -> List[Dict]:
        """
        Get the version history.

        Returns:
            List[Dict]: List of versions
        """
        with st.session_state.thread_lock:
            return st.session_state.protocol_versions.copy()

    def get_version_by_id(self, version_id: str) -> Optional[Dict]:
        """
        Get a specific version by ID.

        Args:
            version_id (str): ID of the version to retrieve

        Returns:
            Optional[Dict]: Version data or None if not found
        """
        with st.session_state.thread_lock:
            for version in st.session_state.protocol_versions:
                if version["id"] == version_id:
                    return version
        return None

    def get_version_by_name(self, version_name: str) -> Optional[Dict]:
        """
        Get a specific version by name.

        Args:
            version_name (str): Name of the version to retrieve

        Returns:
            Optional[Dict]: Version data or None if not found
        """
        with st.session_state.thread_lock:
            for version in st.session_state.protocol_versions:
                if version["name"] == version_name:
                    return version
        return None

    def get_version_by_timestamp(self, timestamp: str) -> Optional[Dict]:
        """
        Get a specific version by timestamp.

        Args:
            timestamp (str): Timestamp of the version to retrieve

        Returns:
            Optional[Dict]: Version data or None if not found
        """
        with st.session_state.thread_lock:
            for version in st.session_state.protocol_versions:
                if version["timestamp"].startswith(timestamp):
                    return version
        return None

    def delete_version(self, version_id: str) -> bool:
        """
        Delete a specific version.

        Args:
            version_id (str): ID of the version to delete

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            with st.session_state.thread_lock:
                st.session_state.protocol_versions = [
                    v
                    for v in st.session_state.protocol_versions
                    if v["id"] != version_id
                ]
                # If we deleted the current version, set to the latest remaining version
                if (
                    st.session_state.get("current_version_id") == version_id
                    and st.session_state.protocol_versions
                ):
                    latest_version = st.session_state.protocol_versions[-1]
                    st.session_state.protocol_text = latest_version["protocol_text"]
                    st.session_state.current_version_id = latest_version["id"]
            return True
        except Exception as e:
            st.error(f"Error deleting version: {e}")
            return False

    def branch_version(self, version_id: str, new_version_name: str) -> Optional[str]:
        """
        Create a new branch from an existing version.

        Args:
            version_id (str): ID of the version to branch from
            new_version_name (str): Name for the new branched version

        Returns:
            Optional[str]: ID of the new version or None if failed
        """
        version = None
        with st.session_state.thread_lock:
            for v in st.session_state.protocol_versions:
                if v["id"] == version_id:
                    version = v
                    break

        if not version:
            st.error("Version not found")
            return None

        # Create new version with branched content
        new_version_id = self.create_new_version(
            version["protocol_text"],
            new_version_name,
            f"Branched from {version['name']}",
        )

        # Add branch metadata
        with st.session_state.thread_lock:
            for v in st.session_state.protocol_versions:
                if v["id"] == new_version_id:
                    v["branch_from"] = version_id
                    v["branch_name"] = new_version_name
                    break

        return new_version_id

    def compare_versions(self, version_id_1: str, version_id_2: str) -> Dict:
        """
        Compare two versions and return the differences.

        Args:
            version_id_1 (str): ID of the first version
            version_id_2 (str): ID of the second version

        Returns:
            Dict: Comparison results with differences
        """
        version1 = self.get_version_by_id(version_id_1)
        version2 = self.get_version_by_id(version_id_2)

        if not version1 or not version2:
            return {"error": "One or both versions not found"}

        # Calculate differences (simplified approach)
        text1 = version1["protocol_text"]
        text2 = version2["protocol_text"]

        # Character-level differences
        chars_added = 0
        chars_removed = 0

        # Find longest common subsequence to estimate differences
        try:
            import difflib

            diff = list(
                difflib.unified_diff(
                    text1.splitlines(keepends=True),
                    text2.splitlines(keepends=True),
                    fromfile=f"Version {version1['name']}",
                    tofile=f"Version {version2['name']}",
                )
            )

            for line in diff:
                if line.startswith("+"):
                    chars_added += len(line) - 1  # -1 for the '+' character
                elif line.startswith("-"):
                    chars_removed += len(line) - 1  # -1 for the '-' character
        except ImportError:
            # Fallback if difflib is not available
            chars_added = abs(len(text2) - len(text1)) if len(text2) > len(text1) else 0
            chars_removed = (
                abs(len(text1) - len(text2)) if len(text1) > len(text2) else 0
            )

        return {
            "version1": version1["name"],
            "version2": version2["name"],
            "chars_added": chars_added,
            "chars_removed": chars_removed,
            "total_chars_change": abs(len(text2) - len(text1)),
            "complexity_diff": {
                "version1": version1["complexity_metrics"],
                "version2": version2["complexity_metrics"],
            },
        }

    def get_version_timeline(self) -> List[Dict]:
        """
        Get a chronological timeline of all versions.

        Returns:
            List[Dict]: Sorted list of versions by timestamp
        """
        with st.session_state.thread_lock:
            versions = st.session_state.protocol_versions.copy()

        # Sort by timestamp
        versions.sort(key=lambda x: x["timestamp"])
        return versions

    def render_version_timeline(self) -> str:
        """
        Render a visual timeline of versions.

        Returns:
            str: HTML formatted timeline
        """
        versions = self.get_version_timeline()

        if not versions:
            return "<p>No version history available</p>"

        html = """
        <div style="background-color: #f8f9fa; padding: 20px; border-radius: 10px;">
            <h3 style="color: #4a6fa5; margin-top: 0;">🕒 Version Timeline</h3>
            <div style="position: relative; padding-left: 30px;">
                <div style="position: absolute; left: 15px; top: 0; bottom: 0; width: 2px; background-color: #4a6fa5;"></div>
        """

        for i, version in enumerate(versions):
            is_current = version["id"] == st.session_state.get("current_version_id", "")
            timestamp = version["timestamp"][:16].replace("T", " ")

            html += f"""
            <div style="position: relative; margin-bottom: 20px;">
                <div style="position: absolute; left: -20px; top: 5px; width: 12px; height: 12px; border-radius: 50%; background-color: {"#4a6fa5" if is_current else "#6b8cbc"}; border: 2px solid white;"></div>
                <div style="background-color: {"#e3f2fd" if is_current else "white"}; padding: 15px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); border-left: 4px solid {"#4a6fa5" if is_current else "#6b8cbc"};">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <h4 style="margin: 0; color: #4a6fa5;">{version["name"]}</h4>
                        <span style="font-size: 0.9em; color: #666;">{timestamp}</span>
                    </div>
                    <p style="margin: 5px 0 0 0; color: #666;">{version.get("comment", "No comment")}</p>
                    <div style="margin-top: 10px; display: flex; gap: 10px;">
            """

            # Add action buttons
            html += f"""
                        <button onclick="loadVersion('{version["id"]}')" style="background-color: #4a6fa5; color: white; border: none; padding: 5px 10px; border-radius: 4px; cursor: pointer; font-size: 0.8em;">Load</button>
            """

            if not is_current:
                html += f"""
                        <button onclick="branchVersion('{version["id"]}', 'Branch of {version["name"]}')" style="background-color: #6b8cbc; color: white; border: none; padding: 5px 10px; border-radius: 4px; cursor: pointer; font-size: 0.8em;">Branch</button>
                """

            html += """
                    </div>
                </div>
            </div>
            """

        html += """
        </div>
    </div>
    <script>
    function loadVersion(versionId) {
        // In a real implementation, this would trigger a reload with the version
        alert('Loading version: ' + versionId);
    }
    
    function branchVersion(versionId, branchName) {
        // In a real implementation, this would create a new branch
        alert('Branching from version: ' + versionId + ' as ' + branchName);
    }
    </script>
    """

        return html

    def get_version_count(self) -> int:
        """
        Get the total number of versions.

        Returns:
            int: Number of versions
        """
        with st.session_state.thread_lock:
            return len(st.session_state.protocol_versions)

    def get_current_version(self) -> Optional[Dict]:
        """
        Get the current version.

        Returns:
            Optional[Dict]: Current version data or None if not found
        """
        current_id = st.session_state.get("current_version_id")
        if current_id:
            return self.get_version_by_id(current_id)
        return None


# Initialize version control on import
version_control = VersionControl()

def render_version_control():
    """
    Placeholder function to render the version control section in the Streamlit UI.
    This would typically allow users to view history, load versions, branch, and compare.
    """
    st.header("📜 Version Control")
    st.info("Version control features are under development. Stay tuned!")
    # Example of how you might use the manager:
    # st.subheader("Version History")
    # version_control.render_version_timeline()
    #
    # if st.button("Create New Version"):
    #     version_control.create_new_version(st.session_state.protocol_text, "Manual Save")
    #     st.success("New version created!")
