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
            "author": st.session_state.get("user", "Unknown User"),
            "complexity_metrics": calculate_protocol_complexity(protocol_text),
            "structure_analysis": extract_protocol_structure(protocol_text),
        }

        with st.session_state.thread_lock:
            st.session_state.protocol_versions.append(version)
            st.session_state.current_version_id = version_id

        return version_id

    function branchVersion(versionId, branchName) {
        const url = new URL(window.location);
        url.searchParams.set('branch_version_id', versionId);
        url.searchParams.set('new_branch_name', branchName);
        window.location.href = url.toString(); // This will trigger a full page reload
    }

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
    Renders the version control section in the Streamlit UI.
    Allows users to view history, load versions, branch, and compare.
    """
    st.header("📜 Version Control")
    
    # Initialize version control if not already done
    vc = VersionControl()

    # Process query parameters for version actions
    query_params = st.experimental_get_query_params()
    
    if "load_version_id" in query_params:
        version_id_to_load = query_params["load_version_id"][0]
        if vc.load_version(version_id_to_load):
            st.success(f"Loaded version: {version_id_to_load[:8]}...")
        else:
            st.error(f"Failed to load version: {version_id_to_load[:8]}...")
        # Clear the query parameter to prevent re-triggering on refresh
        st.experimental_set_query_params(load_version_id=None)
        st.rerun()

    if "branch_version_id" in query_params and "new_branch_name" in query_params:
        version_id_to_branch = query_params["branch_version_id"][0]
        new_branch_name = query_params["new_branch_name"][0]
        new_id = vc.branch_version(version_id_to_branch, new_branch_name)
        if new_id:
            st.success(f"Created branch from {version_id_to_branch[:8]}... as {new_branch_name}")
        else:
            st.error(f"Failed to create branch from {version_id_to_branch[:8]}...")
        # Clear the query parameters
        st.experimental_set_query_params(branch_version_id=None, new_branch_name=None)
        st.rerun()
    
    # Create new version
    with st.expander("💾 Save Current Version"):
        col1, col2 = st.columns([3, 1])
        with col1:
            version_name = st.text_input("Version Name (optional)", placeholder="e.g., Initial Draft, Updated Security Policy")
        with col2:
            st.write("")  # Spacer
            st.write("")  # Spacer
            if st.button("💾 Save Version", use_container_width=True):
                if st.session_state.get("protocol_text", "").strip():
                    comment = st.text_input("Optional Comment", placeholder="Brief description of changes")
                    version_id = vc.create_new_version(st.session_state.protocol_text, version_name or "", comment)
                    if version_id:
                        st.success(f"Version saved successfully with ID: {version_id[:8]}...")
                        st.rerun()
                else:
                    st.error("Cannot save empty protocol text")
    
    # Version history
    st.subheader("📋 Version History")
    versions = vc.get_version_history()
    
    if versions:
        # Show most recent versions first
        versions.reverse()
        
        # Pagination for versions
        items_per_page = 5
        total_pages = (len(versions) + items_per_page - 1) // items_per_page
        if total_pages > 1:
            page = st.selectbox("Page", range(1, total_pages + 1), format_func=lambda x: f"Page {x}")
        else:
            page = 1
            
        start_idx = (page - 1) * items_per_page
        end_idx = start_idx + items_per_page
        current_versions = versions[start_idx:end_idx]
        
        for version in current_versions:
            is_current = version["id"] == st.session_state.get("current_version_id", "")
            with st.container(border=True):
                col1, col2 = st.columns([4, 1])
                with col1:
                    st.write(f"**{version['name']}**")
                    st.caption(f"ID: {version['id'][:8]}...")
                    st.caption(f"Author: {version['author']}")
                    st.caption(f"Date: {version['timestamp'][:19].replace('T', ' ')}")
                    if version.get("comment"):
                        st.text(f"Comment: {version['comment']}")
                with col2:
                    if st.button(f"{'🔴' if is_current else '🔄'} Load", key=f"load_{version['id']}", 
                                type="primary" if is_current else "secondary"):
                        vc.load_version(version["id"])
                        st.success(f"Loaded version: {version['name']}")
                        st.rerun()
                    
                    if not is_current and st.button("SetBranch", key=f"branch_{version['id']}"):
                        new_name = st.text_input("New Branch Name", value=f"Branch of {version['name']}")
                        if st.button("Confirm Branch Creation"):
                            new_id = vc.branch_version(version["id"], new_name)
                            if new_id:
                                st.success(f"Created branch from {version['name']}")
                                st.rerun()
        
        # Compare versions
        st.subheader("🔍 Compare Versions")
        if len(versions) >= 2:
            version_list = [v for v in versions]  # We reversed it earlier, so need to reverse again for proper chronological order
            version_list.reverse()
            
            col1, col2 = st.columns(2)
            with col1:
                version1_id = st.selectbox("First Version", 
                                         [v["id"] for v in version_list], 
                                         format_func=lambda x: next(v["name"] for v in version_list if v["id"] == x))
            with col2:
                version2_id = st.selectbox("Second Version", 
                                         [v["id"] for v in version_list], 
                                         format_func=lambda x: next(v["name"] for v in version_list if v["id"] == x),
                                         index=len(version_list)-1)
            
            if st.button("Compare Versions"):
                comparison = vc.compare_versions(version1_id, version2_id)
                if "error" not in comparison:
                    st.subheader("Comparison Results")
                    st.json(comparison)
                else:
                    st.error(comparison["error"])
        else:
            st.info("Need at least 2 versions to compare")
    else:
        st.info("No versions saved yet. Save your first version above.")
    
    # Current version info
    st.subheader("ℹ️ Current Version Info")
    current = vc.get_current_version()
    if current:
        st.write(f"**Name:** {current['name']}")
        st.write(f"**ID:** {current['id'][:8]}...")
        st.write(f"**Timestamp:** {current['timestamp'][:19].replace('T', ' ')}")
        if current.get("comment"):
            st.write(f"**Comment:** {current['comment']}")
    else:
        st.info("No current version loaded")
