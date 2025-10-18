"""
Template Manager for OpenEvolve - Template marketplace and management
This file manages protocol templates, marketplace, and template-related features
File size: ~900 lines (under the 2000 line limit)
"""

import streamlit as st
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from session_utils import PROTOCOL_TEMPLATE_MARKETPLACE


class TemplateManager:
    """
    Manages protocol templates, marketplace, and template-related features
    """

    def __init__(self):
        self.template_marketplace = PROTOCOL_TEMPLATE_MARKETPLACE
        self.template_usage_db = {
            "Example Template 1": {"times_used": 150, "last_used": "2025-10-15T10:00:00", "user_ratings": [5, 4, 5], "average_rating": 4.6},
            "Example Template 2": {"times_used": 80, "last_used": "2025-10-10T14:30:00", "user_ratings": [3, 4], "average_rating": 3.5},
            "Security Policy Template": {"times_used": 200, "last_used": "2025-10-16T09:00:00", "user_ratings": [5, 5, 4, 5], "average_rating": 4.75},
            "Documentation Template": {"times_used": 50, "last_used": "2025-10-12T11:00:00", "user_ratings": [4], "average_rating": 4.0},
        }

    def list_template_categories(self) -> List[str]:
        """
        List all template categories in the marketplace.

        Returns:
            List[str]: List of category names
        """
        return list(self.template_marketplace.keys())

    def list_templates_in_category(self, category: str) -> List[str]:
        """
        List all templates in a specific category.

        Args:
            category (str): Category name

        Returns:
            List[str]: List of template names
        """
        if category in self.template_marketplace:
            return list(self.template_marketplace[category].keys())
        return []

    def get_template_details(self, category: str, template_name: str) -> Optional[Dict]:
        """
        Get details for a specific template.

        Args:
            category (str): Category name
            template_name (str): Template name

        Returns:
            Optional[Dict]: Template details or None if not found
        """
        if category in self.template_marketplace:
            if template_name in self.template_marketplace[category]:
                return self.template_marketplace[category][template_name]
        return None

    def search_templates(self, query: str) -> List[Tuple[str, str, Dict]]:
        """
        Search templates by query term.

        Args:
            query (str): Search query

        Returns:
            List[Tuple[str, str, Dict]]: List of (category, template_name, details) tuples
        """
        results = []
        query_lower = query.lower()

        for category, templates in self.template_marketplace.items():
            for template_name, details in templates.items():
                # Search in template name, description, tags
                if (
                    query_lower in template_name.lower()
                    or query_lower in details.get("description", "").lower()
                    or any(
                        query_lower in tag.lower() for tag in details.get("tags", [])
                    )
                ):
                    results.append((category, template_name, details))

        # Sort by rating (descending)
        results.sort(key=lambda x: x[2].get("rating", 0), reverse=True)
        return results

    def get_popular_templates(self, limit: int = 10) -> List[Tuple[str, str, Dict]]:
        """
        Get the most popular templates.

        Args:
            limit (int): Maximum number of templates to return

        Returns:
            List[Tuple[str, str, Dict]]: List of (category, template_name, details) tuples
        """
        all_templates = []

        for category, templates in self.template_marketplace.items():
            for template_name, details in templates.items():
                all_templates.append((category, template_name, details))

        # Sort by downloads (descending)
        all_templates.sort(key=lambda x: x[2].get("downloads", 0), reverse=True)
        return all_templates[:limit]

    def get_top_rated_templates(self, limit: int = 10) -> List[Tuple[str, str, Dict]]:
        """
        Get the top-rated templates.

        Args:
            limit (int): Maximum number of templates to return

        Returns:
            List[Tuple[str, str, Dict]]: List of (category, template_name, details) tuples
        """
        all_templates = []

        for category, templates in self.template_marketplace.items():
            for template_name, details in templates.items():
                all_templates.append((category, template_name, details))

        # Sort by rating (descending)
        all_templates.sort(key=lambda x: x[2].get("rating", 0), reverse=True)
        return all_templates[:limit]

    def render_template_marketplace_ui(self) -> str:
        """
        Render the template marketplace UI.

        Returns:
            str: HTML formatted marketplace UI
        """
        html = """
        <div style="background-color: #f8f9fa; padding: 20px; border-radius: 10px; margin-bottom: 20px;">
            <h2 style="color: #4a6fa5; margin-top: 0; text-align: center;">🛍️ Protocol Template Marketplace</h2>
            
            <!-- Search Bar -->
            <div style="margin-bottom: 20px;">
                <input type="text" id="templateSearch" placeholder="Search templates..." style="width: 100%; padding: 10px; border-radius: 5px; border: 1px solid #ddd;">
            </div>
            
            <!-- Quick Filters -->
            <div style="display: flex; gap: 10px; margin-bottom: 20px; flex-wrap: wrap;">
                <button onclick="filterTemplates('all')" style="background-color: #4a6fa5; color: white; border: none; padding: 8px 15px; border-radius: 20px; cursor: pointer;">All Templates</button>
                <button onclick="filterTemplates('popular')" style="background-color: #6b8cbc; color: white; border: none; padding: 8px 15px; border-radius: 20px; cursor: pointer;">Most Popular</button>
                <button onclick="filterTemplates('rated')" style="background-color: #8ca7d1; color: white; border: none; padding: 8px 15px; border-radius: 20px; cursor: pointer;">Top Rated</button>
            </div>
            
            <!-- Template Grid -->
            <div id="templateGrid" style="display: grid; grid-template-columns: repeat(auto-fill, minmax(300px, 1fr)); gap: 20px;">
        """

        # Add featured templates
        popular_templates = self.get_popular_templates(6)
        for category, template_name, details in popular_templates:
            html += f"""
            <div style="background-color: white; border-radius: 10px; padding: 15px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); border: 1px solid #eee;">
                <h3 style="color: #4a6fa5; margin-top: 0;">{template_name}</h3>
                <p style="color: #666; font-size: 0.9em;">{details.get("description", "")[:100]}...</p>
                <div style="display: flex; justify-content: space-between; align-items: center; margin-top: 10px;">
                    <span style="background-color: #e3f2fd; color: #1565c0; padding: 3px 8px; border-radius: 10px; font-size: 0.8em;">
                        {details.get("category", "Uncategorized")}
                    </span>
                    <div style="text-align: right;">
                        <div style="color: #ffa000;">{"★" * int(details.get("rating", 0) // 2)} ({details.get("rating", 0)})</div>
                        <div style="font-size: 0.8em; color: var(--text-light);">{details.get("downloads", 0):,} downloads</div>
                    </div>
                </div>
                <div style="margin-top: 10px;">
                    <button onclick="loadTemplate('{category}', '{template_name}')" style="width: 100%; background-color: #4a6fa5; color: white; border: none; padding: 8px; border-radius: 5px; cursor: pointer;">
                        Load Template
                    </button>
                </div>
            </div>
            """

        html += """
            </div>
        </div>
        
        <script>
        function loadTemplate(category, templateName) {
            const url = new URL(window.location);
            url.searchParams.set('load_template_category', category);
            url.searchParams.set('load_template_name', templateName);
            window.location.href = url.toString(); // This will trigger a full page reload
        }
        
        function filterTemplates(filterType) {
            const url = new URL(window.location);
            url.searchParams.set('filter_type', filterType);
            window.location.href = url.toString(); // This will trigger a full page reload
        }
        
        document.getElementById('templateSearch').addEventListener('change', function(e) { // Use 'change' instead of 'input' to trigger on blur or enter
            const url = new URL(window.location);
            url.searchParams.set('template_search_query', e.target.value);
            window.location.href = url.toString(); // This will trigger a full page reload
        });
        </script>
        """

        return html

    def add_custom_template(
        self,
        category: str,
        template_name: str,
        template_content: str,
        description: str = "",
        tags: List[str] = None,
    ) -> bool:
        """
        Add a custom template to the system.

        Args:
            category (str): Category for the template
            template_name (str): Name of the template
            template_content (str): Content of the template
            description (str): Description of the template
            tags (List[str]): List of tags for the template

        Returns:
            bool: True if successful, False otherwise
        """
        if tags is None:
            tags = []

        try:
            # Create the new template
            new_template = {
                "description": description,
                "category": category,
                "complexity": "Custom",
                "compliance": [],
                "tags": tags,
                "author": "Custom",
                "rating": 0.0,
                "downloads": 0,
                "content": template_content,
                "created_at": datetime.now().isoformat(),
            }

            # Add to session state for custom templates
            if "custom_templates" not in st.session_state:
                st.session_state.custom_templates = {}

            if category not in st.session_state.custom_templates:
                st.session_state.custom_templates[category] = {}

            st.session_state.custom_templates[category][template_name] = new_template
            return True
        except Exception as e:
            st.error(f"Error adding custom template: {e}")
            return False

    def get_all_templates(self) -> Dict[str, Dict[str, Dict]]:
        """
        Get all templates (marketplace + custom).

        Returns:
            Dict[str, Dict[str, Dict]]: All templates organized by category
        """
        all_templates = self.template_marketplace.copy()

        # Add custom templates if they exist
        if "custom_templates" in st.session_state:
            for category, templates in st.session_state.custom_templates.items():
                if category not in all_templates:
                    all_templates[category] = {}
                all_templates[category].update(templates)

        return all_templates

    def get_template_usage_stats(self, template_name: str) -> Dict:
        """
        Get usage statistics for a template.

        Args:
            template_name (str): Name of the template

        Returns:
            Dict: Usage statistics
        """
        # This would connect to a database in a full implementation
        # For now, we use a simulated database for usage statistics
        return self.template_usage_db.get(template_name, {
            "times_used": 0,
            "last_used": None,
            "user_ratings": [],
            "average_rating": 0.0,
        })


# Initialize template manager on import
template_manager = TemplateManager()

def render_template_manager():
    """
    Renders the template manager section in the Streamlit UI.
    Allows users to browse, search, and manage templates.
    """
    st.header("📚 Template Manager")
    
    # Initialize template manager
    tm = TemplateManager()

    # Process query parameters for template actions
    query_params = st.experimental_get_query_params()

    if "load_template_category" in query_params and "load_template_name" in query_params:
        category = query_params["load_template_category"][0]
        template_name = query_params["load_template_name"][0]
        details = tm.get_template_details(category, template_name)
        if details:
            st.session_state.protocol_text = details.get("content", "")
            st.success(f"Template '{template_name}' loaded!")
        else:
            st.error(f"Failed to load template: {template_name}")
        st.experimental_set_query_params(load_template_category=None, load_template_name=None)
        st.rerun()

    # Note: filter_type and template_search_query will be handled by the Streamlit UI elements directly
    # as they are now triggering reruns and the UI elements will pick up the new state.
    # We don't need explicit processing here for them, but we should clear them if they exist
    # to prevent persistent filtering/searching across sessions if not desired.
    if "filter_type" in query_params:
        st.experimental_set_query_params(filter_type=None)
        # st.rerun() # No need to rerun again, the main UI will handle the filtering based on the new state

    if "template_search_query" in query_params:
        st.experimental_set_query_params(template_search_query=None)
        # st.rerun() # No need to rerun again
    
    # Create tabs for different template functions
    tab1, tab2, tab3 = st.tabs(["🏪 Marketplace", "📁 My Templates", "➕ Create Template"])
    
    with tab1:
        st.subheader("Template Marketplace")
        
        # Search functionality
        search_query = st.text_input("Search Templates", placeholder="Search by name, description, or tags...")
        
        if search_query:
            results = tm.search_templates(search_query)
            if results:
                for category, template_name, details in results:
                    with st.container(border=True):
                        col1, col2 = st.columns([4, 1])
                        with col1:
                            st.write(f"**{template_name}**")
                            st.caption(f"Category: {details.get('category', 'General')}")
                            st.caption(details.get("description", "No description provided"))
                            if details.get("tags"):
                                tag_str = " ".join([f"`{tag}`" for tag in details.get("tags", [])[:3]])  # Show first 3 tags
                                st.caption(f"Tags: {tag_str}")
                        with col2:
                            st.caption(f"⭐ {details.get('rating', 0)}/5 ({details.get('downloads', 0)} downloads)")
                            if st.button("Use Template", key=f"use_{template_name}"):
                                st.session_state.protocol_text = details.get("content", "")
                                st.success(f"Template '{template_name}' loaded!")
                                st.rerun()
        else:
            # Show categories and templates
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Popular Templates**")
                popular = tm.get_popular_templates(5)
                for category, template_name, details in popular:
                    if st.button(f"📋 {template_name}", key=f"popular_{template_name}"):
                        st.session_state.protocol_text = details.get("content", "")
                        st.success(f"Loaded popular template: {template_name}")
                        st.rerun()
            
            with col2:
                st.write("**Top Rated Templates**")
                top_rated = tm.get_top_rated_templates(5)
                for category, template_name, details in top_rated:
                    if st.button(f"⭐ {template_name}", key=f"rated_{template_name}"):
                        st.session_state.protocol_text = details.get("content", "")
                        st.success(f"Loaded top-rated template: {template_name}")
                        st.rerun()
            
            # Show by category
            st.subheader("Browse by Category")
            categories = tm.list_template_categories()
            if categories:
                selected_category = st.selectbox("Select Category", categories)
                if selected_category:
                    templates = tm.list_templates_in_category(selected_category)
                    if templates:
                        for template_name in templates:
                            details = tm.get_template_details(selected_category, template_name)
                            with st.container(border=True):
                                st.write(f"**{template_name}**")
                                st.caption(details.get("description", "No description provided"))
                                col1, col2 = st.columns([3, 1])
                                with col1:
                                    st.caption(f"⭐ Rating: {details.get('rating', 0)}/5 | 📥 Downloads: {details.get('downloads', 0)}")
                                with col2:
                                    if st.button("Load", key=f"load_{template_name}"):
                                        st.session_state.protocol_text = details.get("content", "")
                                        st.success(f"Template '{template_name}' loaded!")
                                        st.rerun()
    
    with tab2:
        st.subheader("My Custom Templates")
        
        # Show custom templates if any exist
        
        custom_templates_exist = "custom_templates" in st.session_state and st.session_state.get("custom_templates", {})
        
        if custom_templates_exist:
            custom_cats = st.session_state.custom_templates
            for category, templates in custom_cats.items():
                st.write(f"**{category}**")
                for template_name, details in templates.items():
                    with st.container(border=True):
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            st.write(f"**{template_name}**")
                            st.caption(details.get("description", "No description"))
                        with col2:
                            if st.button("Use", key=f"my_{template_name}"):
                                st.session_state.protocol_text = details.get("content", "")
                                st.success(f"Loaded custom template: {template_name}")
                                st.rerun()
        else:
            st.info("You don't have any custom templates yet. Create one in the 'Create Template' tab!")
    
    with tab3:
        st.subheader("Create New Template")
        
        with st.form("create_template_form"):
            col1, col2 = st.columns(2)
            with col1:
                new_template_name = st.text_input("Template Name", placeholder="e.g., Security Policy Template")
                category = st.text_input("Category", placeholder="e.g., Security, Documentation")
            with col2:
                description = st.text_input("Description", placeholder="Brief description of the template")
            
            tags = st.text_input("Tags (comma-separated)", placeholder="e.g., security, policy, compliance")
            template_content = st.text_area("Template Content", height=200, 
                                          value=st.session_state.get("protocol_text", ""))
            
            submitted = st.form_submit_button("Save as Template")
            if submitted:
                if new_template_name.strip() and template_content.strip():
                    tags_list = [tag.strip() for tag in tags.split(",")] if tags.strip() else []
                    if tm.add_custom_template(category or "Uncategorized", 
                                            new_template_name, 
                                            template_content, 
                                            description, 
                                            tags_list):
                        st.success(f"Template '{new_template_name}' saved successfully!")
                        st.rerun()
                    else:
                        st.error("Failed to save template")
                else:
                    st.error("Template name and content are required!")
        
        st.info("💡 Pro Tip: You can save your current content as a template to reuse later!")
