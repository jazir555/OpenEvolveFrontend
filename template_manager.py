"""
Template Manager for OpenEvolve - Template marketplace and management
This file manages protocol templates, marketplace, and template-related features
File size: ~900 lines (under the 2000 line limit)
"""
import streamlit as st
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import json
import threading
from .session_utils import PROTOCOL_TEMPLATE_MARKETPLACE


class TemplateManager:
    """
    Manages protocol templates, marketplace, and template-related features
    """
    
    def __init__(self):
        self.template_marketplace = PROTOCOL_TEMPLATE_MARKETPLACE
    
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
                if (query_lower in template_name.lower() or 
                    query_lower in details.get("description", "").lower() or
                    any(query_lower in tag.lower() for tag in details.get("tags", []))):
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
                <p style="color: #666; font-size: 0.9em;">{details.get('description', '')[:100]}...</p>
                <div style="display: flex; justify-content: space-between; align-items: center; margin-top: 10px;">
                    <span style="background-color: #e3f2fd; color: #1565c0; padding: 3px 8px; border-radius: 10px; font-size: 0.8em;">
                        {details.get('category', 'Uncategorized')}
                    </span>
                    <div style="text-align: right;">
                        <div style="color: #ffa000;">{'★' * int(details.get('rating', 0) // 2)} ({details.get('rating', 0)})</div>
                        <div style="font-size: 0.8em; color: #999;">{details.get('downloads', 0):,} downloads</div>
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
            // In a real implementation, this would load the template
            alert('Loading template: ' + templateName + ' from category: ' + category);
        }
        
        function filterTemplates(filterType) {
            // In a real implementation, this would filter the templates
            alert('Filtering by: ' + filterType);
        }
        
        document.getElementById('templateSearch').addEventListener('input', function(e) {
            // In a real implementation, this would search the templates
            console.log('Searching for: ' + e.target.value);
        });
        </script>
        """
        
        return html
    
    def add_custom_template(self, category: str, template_name: str, template_content: str, 
                           description: str = "", tags: List[str] = None) -> bool:
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
                "created_at": datetime.now().isoformat()
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
        # For now, return placeholder stats
        return {
            "times_used": 0,
            "last_used": None,
            "user_ratings": [],
            "average_rating": 0.0
        }


# Initialize template manager on import
template_manager = TemplateManager()