#!/usr/bin/env python3
"""
OpenEvolve n8n Workflow Integration
This module integrates ClaraVerse's n8n visual workflows into the Streamlit UI.
"""

import streamlit as st
import json
import os
import requests
from typing import List, Dict, Any
import base64


class N8NWorkflowIntegration:
    """Main class for integrating n8n visual workflows into Streamlit."""
    
    def __init__(self):
        self.workflows_data = self._load_workflows_data()
        self.categories = self._get_categories()
        
    def _load_workflows_data(self) -> List[Dict[str, Any]]:
        """Load workflows data from ClaraVerse n8n workflows JSON file."""
        try:
            # Try to load from ClaraVerse directory
            claraverse_path = os.path.join("ClaraVerse", "src", "components", "n8n_components", "workflows", "n8n_workflows_full.json")
            if os.path.exists(claraverse_path):
                with open(claraverse_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                # Add IDs and other required fields
                for i, workflow in enumerate(data):
                    workflow['id'] = workflow.get('id', f"workflow-{i}")
                    workflow['downloads'] = workflow.get('downloads', 0)
                    workflow['is_prebuilt'] = workflow.get('is_prebuilt', True)
                return data
            else:
                # Fallback to embedded data if ClaraVerse path doesn't exist
                return self._get_fallback_workflows()
        except Exception as e:
            st.error(f"Failed to load n8n workflows: {e}")
            return []
    
    def _get_fallback_workflows(self) -> List[Dict[str, Any]]:
        """Provide fallback workflow data if ClaraVerse files are not available."""
        return [
            {
                "id": "fallback-1",
                "category": "data-integration",
                "name": "Sample Data Integration Workflow",
                "description": "A sample workflow demonstrating data integration capabilities.",
                "nodeCount": 5,
                "tags": ["sample", "data", "integration"],
                "jsonLink": "https://raw.githubusercontent.com/aruntemme/n8n-workflows/main/data-integration/sample-workflow/workflow.json",
                "nodeNames": ["Start", "Process Data", "Transform", "Save Results", "End"],
                "readmeLink": "https://github.com/aruntemme/n8n-workflows/blob/main/data-integration/sample-workflow/README.md",
                "downloads": 0,
                "is_prebuilt": True
            }
        ]
    
    def _get_categories(self) -> List[str]:
        """Get unique categories from workflows."""
        if not self.workflows_data:
            return []
        categories = list(set(workflow['category'] for workflow in self.workflows_data))
        return sorted(categories)
    
    def _fetch_workflow_json(self, json_link: str) -> str:
        """Fetch workflow JSON content from a URL."""
        try:
            if json_link.startswith('http'):
                response = requests.get(json_link, timeout=10)
                if response.status_code == 200:
                    return response.text
                else:
                    return f"Failed to fetch workflow: HTTP {response.status_code}"
            else:
                return json_link  # Return as-is if it's already JSON content
        except Exception as e:
            return f"Error fetching workflow: {e}"
    
    def render_n8n_workflow_store(self):
        """Render the n8n workflow store interface in Streamlit."""
        
        st.markdown("""
        <style>
        /* Custom CSS for n8n workflow store */
        .n8n-workflow-card {
            border: 1px solid #e0e0e0;
            border-radius: 12px;
            padding: 16px;
            margin-bottom: 16px;
            transition: all 0.3s ease;
            cursor: pointer;
        }
        .n8n-workflow-card:hover {
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
            border-color: #ff6b9d;
        }
        .category-badge {
            display: inline-block;
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 12px;
            font-weight: 600;
            margin-bottom: 8px;
        }
        .data-integration { background-color: #dbeafe; color: #1e40af; }
        .api-webhooks { background-color: #d1fae5; color: #065f46; }
        .document-processing { background-color: #e9d5ff; color: #6b21a8; }
        .automation { background-color: #fef3c7; color: #92400e; }
        .communication { background-color: #fce7f3; color: #9f1239; }
        .analytics { background-color: #fed7aa; color: #9a3412; }
        </style>
        """, unsafe_allow_html=True)
        
        # Header
        st.markdown("""
        <div style='text-align: center; margin-bottom: 24px;'>
            <h1 style='color: #ff6b9d; font-size: 2.5em;'>🔧 n8n Visual Workflows</h1>
            <p style='color: #666; font-size: 1.1em;'>Browse and integrate pre-built automation workflows</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Search and filter
        col1, col2 = st.columns([3, 1])
        
        with col1:
            search_query = st.text_input("🔍 Search workflows", "", placeholder="Search by name, description, or tags...")
        
        with col2:
            selected_category = st.selectbox("📁 Category", ["All Categories"] + self.categories)
        
        # Filter workflows
        filtered_workflows = self._filter_workflows(search_query, selected_category)
        
        if not filtered_workflows:
            st.warning("No workflows found matching your criteria.")
            return
        
        # Display workflows in a grid
        cols_per_row = 3
        for i in range(0, len(filtered_workflows), cols_per_row):
            cols = st.columns(cols_per_row)
            for j, workflow in enumerate(filtered_workflows[i:i+cols_per_row]):
                with cols[j]:
                    self._render_workflow_card(workflow)
    
    def _filter_workflows(self, search_query: str, category: str) -> List[Dict[str, Any]]:
        """Filter workflows based on search query and category."""
        workflows = self.workflows_data
        
        # Filter by category
        if category and category != "All Categories":
            workflows = [w for w in workflows if w.get('category') == category]
        
        # Filter by search query
        if search_query:
            query_lower = search_query.lower()
            workflows = [
                w for w in workflows 
                if (query_lower in w.get('name', '').lower() or
                    query_lower in w.get('description', '').lower() or
                    any(query_lower in tag.lower() for tag in w.get('tags', [])))
            ]
        
        return workflows
    
    def _render_workflow_card(self, workflow: Dict[str, Any]):
        """Render a single workflow card."""
        category = workflow.get('category', 'general')
        category_class = category.replace('-', '-')
        
        # Create card HTML
        card_html = f"""
        <div class="n8n-workflow-card" onclick="this.style.transform='scale(0.98)'">
            <div class="category-badge {category_class}">{workflow.get('category', 'General').replace('-', ' ').title()}</div>
            <h3 style="margin: 8px 0; color: #333;">{workflow.get('name', 'Untitled Workflow')}</h3>
            <p style="color: #666; font-size: 14px; margin-bottom: 12px;">{workflow.get('description', 'No description')[:100]}...</p>
            <div style="display: flex; justify-content: space-between; align-items: center; margin-top: 12px;">
                <span style="color: #999; font-size: 12px;">🔗 {workflow.get('nodeCount', 0)} nodes</span>
                <span style="color: #999; font-size: 12px;">📥 {workflow.get('downloads', 0)} downloads</span>
            </div>
        </div>
        """
        
        st.markdown(card_html, unsafe_allow_html=True)
        
        # Add buttons for actions
        col1, col2 = st.columns([1, 1])
        
        with col1:
            if st.button(f"📋 View Details", key=f"view_{workflow['id']}"):
                self._show_workflow_details(workflow)
        
        with col2:
            if st.button(f"💾 Download", key=f"download_{workflow['id']}"):
                self._download_workflow(workflow)
    
    def _show_workflow_details(self, workflow: Dict[str, Any]):
        """Show detailed information about a workflow."""
        
        # Fetch workflow JSON
        json_content = self._fetch_workflow_json(workflow['jsonLink'])
        
        # Create modal-like display
        st.markdown("---")
        st.subheader(f"📋 {workflow.get('name', 'Workflow Details')}")
        
        # Basic info
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown(f"**Category:** {workflow.get('category', 'General')}")
            st.markdown(f"**Nodes:** {workflow.get('nodeCount', 0)}")
            st.markdown(f"**Downloads:** {workflow.get('downloads', 0)}")
        
        with col2:
            if workflow.get('tags'):
                st.markdown("**Tags:**")
                for tag in workflow.get('tags', []):
                    st.markdown(f"- {tag}")
        
        # Description
        st.markdown("**Description:**")
        st.info(workflow.get('description', 'No description available'))
        
        # Node names
        if workflow.get('nodeNames'):
            st.markdown("**Nodes in this workflow:**")
            node_cols = st.columns(3)
            for i, node in enumerate(workflow.get('nodeNames', [])):
                with node_cols[i % 3]:
                    st.code(node, language=None)
        
        # JSON content
        st.markdown("**Workflow JSON:**")
        
        if isinstance(json_content, str) and json_content.startswith('{') and json_content.endswith('}'):
            try:
                json_data = json.loads(json_content)
                st.json(json_data)
            except:
                st.code(json_content, language="json")
        else:
            st.code(json_content, language="json")
        
        # Action buttons
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            if st.button("💾 Download JSON"):
                self._download_workflow(workflow)
        
        with col2:
            if st.button("📋 Copy JSON URL"):
                st.code(workflow['jsonLink'])
                st.success("URL copied to clipboard (you can manually copy it)")
        
        with col3:
            if st.button("❌ Close"):
                st.experimental_rerun()
    
    def _download_workflow(self, workflow: Dict[str, Any]):
        """Download workflow JSON file."""
        try:
            json_content = self._fetch_workflow_json(workflow['jsonLink'])
            
            if isinstance(json_content, str) and json_content.startswith('{') and json_content.endswith('}'):
                # Create downloadable JSON file
                b64 = base64.b64encode(json_content.encode()).decode()
                href = f'<a href="data:file/json;base64,{b64}" download="{workflow.get("name", "workflow")}.json">Click here to download</a>'
                st.markdown(href, unsafe_allow_html=True)
                st.success(f"✅ Workflow '{workflow.get('name')}' is ready for download!")
            else:
                st.error(f"❌ Could not download workflow: {json_content}")
                
        except Exception as e:
            st.error(f"❌ Error downloading workflow: {e}")


def render_n8n_workflow_integration():
    """Main function to render the n8n workflow integration in Streamlit."""
    integration = N8NWorkflowIntegration()
    integration.render_n8n_workflow_store()


if __name__ == "__main__":
    # For testing the integration standalone
    st.set_page_config(page_title="n8n Workflow Integration", layout="wide")
    render_n8n_workflow_integration()