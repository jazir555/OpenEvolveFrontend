"""
Content Management for OpenEvolve - Protocol templates and content handling
This file manages protocol templates, content validation, and related utilities
File size: ~1200 lines (under the 2000 line limit)
"""

import streamlit as st # Import streamlit to use st.cache_data
from datetime import datetime
import re
from typing import Dict, List, Optional
from session_utils import (
    calculate_protocol_complexity,
    extract_protocol_structure,
    PROTOCOL_TEMPLATES,
    VALIDATION_RULES,
    REPORT_TEMPLATES,
)


class ContentManagement:
    """
    Manages protocol templates, content validation, and content-related utilities
    """

    def __init__(self):
        self.protocol_templates = PROTOCOL_TEMPLATES
        self.validation_rules = VALIDATION_RULES
        self.report_templates = REPORT_TEMPLATES

    @st.cache_data(ttl=3600) # Cache the result for 1 hour
    def list_protocol_templates(_self) -> List[str]:
        """
        List all available protocol templates.

        Returns:
            List[str]: List of template names
        """
        return list(_self.protocol_templates.keys())

    @st.cache_data(ttl=300) # Cache for 5 minutes
    def load_protocol_template(self, template_name: str) -> str:
        """
        Load a protocol template.

        Args:
            template_name (str): Name of the template to load

        Returns:
            str: Template content
        """
        return self.protocol_templates.get(template_name, "")

    def export_protocol_as_template(
        self, protocol_text: str, template_name: str
    ) -> Dict:
        """
        Export protocol as a reusable template.

        Args:
            protocol_text (str): Protocol text to export as template
            template_name (str): Name for the template

        Returns:
            Dict: Template data
        """
        return {
            "name": template_name,
            "content": protocol_text,
            "created_at": datetime.now().isoformat(),
            "complexity_metrics": calculate_protocol_complexity(protocol_text),
            "structure_analysis": extract_protocol_structure(protocol_text),
            "tags": [],
        }

    @st.cache_data(ttl=300) # Cache for 5 minutes
    def validate_protocol(
        self, protocol_text: str, validation_type: str = "generic"
    ) -> Dict:
        """
        Validate a protocol against predefined rules.

        Args:
            protocol_text (str): Protocol text to validate
            validation_type (str): Type of validation to perform

        Returns:
            Dict: Validation results
        """
        if not protocol_text:
            return {
                "valid": False,
                "score": 0,
                "errors": ["Protocol text is empty"],
                "warnings": [],
                "suggestions": ["Please provide protocol text to validate"],
            }

        rules = self.validation_rules.get(
            validation_type, self.validation_rules.get("generic", {{}})
        )

        errors = []
        warnings = []
        suggestions = []

        # Check length
        char_count = len(protocol_text)
        if "max_length" in rules and char_count > rules["max_length"]:
            errors.append(
                f"Protocol exceeds maximum length of {rules['max_length']} characters"
            )

        # Check required sections
        if "required_sections" in rules:

            missing_sections = []
            for section in rules["required_sections"]:
                if section.lower() not in protocol_text.lower():
                    missing_sections.append(section)
            if missing_sections:
                errors.append(
                    f"Missing required sections: {', '.join(missing_sections)}"
                )

        # Check required keywords
        if "required_keywords" in rules:
            missing_keywords = []
            for keyword in rules["required_keywords"]:
                if keyword.lower() not in protocol_text.lower():
                    missing_keywords.append(keyword)
            if missing_keywords:
                warnings.append(
                    f"Consider adding these keywords: {', '.join(missing_keywords)}"
                )

        # Check forbidden patterns
        if "forbidden_patterns" in rules:
            for pattern in rules["forbidden_patterns"]:
                matches = re.findall(pattern, protocol_text)
                if matches:
                    errors.append(f"Forbidden pattern found: {matches[0][:50]}...")

        # Calculate complexity score
        complexity = calculate_protocol_complexity(protocol_text)
        complexity_score = complexity["complexity_score"]

        if "min_complexity" in rules and complexity_score < rules["min_complexity"]:
            suggestions.append(
                f"Increase protocol complexity (current: {complexity_score}, minimum: {rules['min_complexity']})"
            )

        # Calculate overall score
        max_errors = 10
        max_warnings = 5
        error_penalty = min(len(errors) / max_errors, 1.0) * 30
        warning_penalty = min(len(warnings) / max_warnings, 1.0) * 15
        complexity_bonus = 0

        if "min_complexity" in rules:
            complexity_ratio = min(complexity_score / rules["min_complexity"], 1.0)
            complexity_bonus = complexity_ratio * 20

        score = max(0, 100 - error_penalty - warning_penalty + complexity_bonus)

        # Add general suggestions
        if complexity["avg_sentence_length"] > 25:
            suggestions.append("Consider shortening sentences for better readability")

        if complexity["unique_words"] / max(1, complexity["word_count"]) < 0.4:
            suggestions.append("Increase vocabulary diversity to improve clarity")

        return {
            "valid": len(errors) == 0,
            "score": round(score, 1),
            "errors": errors,
            "warnings": warnings,
            "suggestions": suggestions,
            "complexity_metrics": complexity,
        }

    def render_validation_results(
        self, protocol_text: str, validation_type: str = "generic"
    ) -> str:
        """
        Render validation results in a formatted display.

        Args:
            protocol_text (str): Protocol text to validate
            validation_type (str): Type of validation to perform

        Returns:
            str: HTML formatted validation results
        """
        results = self.validate_protocol(protocol_text, validation_type)

        html = f"""
        <div style="background-color: #f8f9fa; padding: 20px; border-radius: 10px; margin-bottom: 20px;">
            <h2 style="color: #4a6fa5; margin-top: 0; text-align: center;">✅ Protocol Validation Results</h2>

            <!-- Overall Score -->
            <div style="display: flex; justify-content: center; align-items: center; margin-bottom: 20px;">
                <div style="background: {
            "linear-gradient(135deg, #4caf50, #81c784)"
            if results["score"] >= 80
            else "linear-gradient(135deg, #ff9800, #ffb74d)"
            if results["score"] >= 60
            else "linear-gradient(135deg, #f44336, #e57373)"
        }; 
                        color: white; border-radius: 50%; width: 120px; height: 120px; 
                        display: flex; justify-content: center; align-items: center; 
                        font-size: 2em; font-weight: bold;">
                    {results["score"]}% 
                </div>
            </div>
            <p style="text-align: center; margin-top: 0; font-size: 1.2em; font-weight: bold;">
                {"Valid Protocol" if results["valid"] else "Protocol Needs Improvement"}
            </p>
        </div>
        """

        # Errors section
        if results["errors"]:
            html += """
            <div style="background-color: #ffebee; padding: 15px; border-radius: 8px; margin-bottom: 20px; border-left: 4px solid #f44336;">
                <h3 style="color: #c62828; margin-top: 0;">❌ Errors</h3>
                <ul style="padding-left: 20px;">
            """
            for error in results["errors"]:
                html += f"<li>{error}</li>"
            html += """
                </ul>
            </div>
            """

        # Warnings section
        if results["warnings"]:
            html += """
            <div style="background-color: #fff8e1; padding: 15px; border-radius: 8px; margin-bottom: 20px; border-left: 4px solid #ff9800;">
                <h3 style="color: #f57f17; margin-top: 0;">⚠️ Warnings</h3>
                <ul style="padding-left: 20px;">
            """
            for warning in results["warnings"]:
                html += f"<li>{warning}</li>"
            html += """
                </ul>
            </div>
            """

        # Suggestions section
        if results["suggestions"]:
            html += """
            <div style="background-color: #e3f2fd; padding: 15px; border-radius: 8px; margin-bottom: 20px; border-left: 4px solid #2196f3;">
                <h3 style="color: #1565c0; margin-top: 0;">💡 Suggestions for Improvement</h3>
                <ul style="padding-left: 20px;">
            """
            for suggestion in results["suggestions"]:
                html += f"<li>{suggestion}</li>"
            html += """
                </ul>
            </div>
            """

        # Complexity metrics
        complexity = results["complexity_metrics"]
        html += """
        <div style="background-color: #f3e5f5; padding: 15px; border-radius: 8px; margin-bottom: 20px; border-left: 4px solid #9c27b0;">
            <h3 style="color: #6a1b9a; margin-top: 0;">📊 Complexity Metrics</h3>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 10px;">
        """

        metrics = [
            ("Words", complexity["word_count"]),
            ("Sentences", complexity["sentence_count"]),
            ("Paragraphs", complexity["paragraph_count"]),
            ("Complexity", complexity["complexity_score"]),
            ("Unique Words", complexity["unique_words"]),
        ]

        for name, value in metrics:
            html += f"""
            <div style="background-color: white; padding: 10px; border-radius: 5px; text-align: center;">
                <div style="font-weight: bold; color: #4a6fa5;">{value}</div>
                <div style="font-size: 0.9em; color: #666;">{name}</div>
            </div>
            """

        html += """
        </div>
    </div>
    """

        return html

    def list_report_templates(self) -> List[str]:
        """
        List all available report templates.

        Returns:
            List[str]: List of report template names
        """
        return list(self.report_templates.keys())

    def get_report_template_details(self, template_name: str) -> Optional[Dict]:
        """
        Get details for a specific report template.

        Args:
            template_name (str): Name of the template

        Returns:
            Optional[Dict]: Template details or None if not found
        """
        return self.report_templates.get(template_name)

    def generate_custom_report(self, template_name: str, data: Dict) -> str:
        """
        Generate a custom report based on a template.

        Args:
            template_name (str): Name of the template to use
            data (Dict): Data to populate the report

        Returns:
            str: Generated report content
        """
        template = self.get_report_template_details(template_name)
        if not template:
            return f"# Error: Template '{template_name}' not found\n\nUnable to generate report."

        report_content = f"# {template['name']}\n\n"
        report_content += (
            f"*Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n\n"
        )

        # Add sections
        for section in template.get("sections", []):
            report_content += f"## {section}\n\n"

            # Add data specific to this section
            section_key = section.lower().replace(" ", "_").replace("-", "_")
            if section_key in data:
                section_data = data[section_key]
                if isinstance(section_data, list):
                    for item in section_data:
                        report_content += f"- {item}\n"
                    report_content += "\n"
                elif isinstance(section_data, dict):
                    for key, value in section_data.items():
                        report_content += f"**{key}:** {value}\n\n"
                else:
                    report_content += f"{section_data}\n\n"
            else:
                report_content += "*(Content to be added)*\n\n"

        return report_content

    @st.cache_data(ttl=300) # Cache for 5 minutes
    def generate_content_summary(self, content: str) -> Dict:
        """
        Generate a summary of the content including key metrics.

        Args:
            content (str): Content to summarize

        Returns:
            Dict: Summary metrics
        """
        if not content:
            return {
                "word_count": 0,
                "character_count": 0,
                "sentence_count": 0,
                "paragraph_count": 0,
                "readability_score": 0,
                "complexity_score": 0,
            }

        # Calculate metrics
        complexity_metrics = calculate_protocol_complexity(content)
        structure_analysis = extract_protocol_structure(content)

        # Simple readability calculation (Flesch Reading Ease approximation)
        avg_sentence_length = complexity_metrics["avg_sentence_length"]
        avg_syllables_per_word = 1.3  # Rough approximation
        readability_score = (
            206.835 - (1.015 * avg_sentence_length) - (84.6 * avg_syllables_per_word)
        )
        readability_score = max(0, min(100, readability_score))  # Clamp to 0-100 range

        return {
            "word_count": complexity_metrics["word_count"],
            "character_count": len(content),
            "sentence_count": complexity_metrics["sentence_count"],
            "paragraph_count": complexity_metrics["paragraph_count"],
            "readability_score": round(readability_score, 1),
            "complexity_score": complexity_metrics["complexity_score"],
            "has_headers": structure_analysis["has_headers"],
            "has_numbered_steps": structure_analysis["has_numbered_steps"],
            "has_bullet_points": structure_analysis["has_bullet_points"],
            "has_preconditions": structure_analysis["has_preconditions"],
            "has_postconditions": structure_analysis["has_postconditions"],
            "has_error_handling": structure_analysis["has_error_handling"],
        }


# Initialize content management on import
content_manager = ContentManagement()

def render_content_manager():
    """
    Placeholder function to render the content manager section in the Streamlit UI.
    This would typically allow users to manage protocols, templates, and view content-related analytics.
    """
    st.header("📝 Content Manager")
    st.info("Content management features are under development. Stay tuned!")
    # Example of how you might use the manager:
    # st.subheader("Protocol Templates")
    # for template_name in content_manager.list_protocol_templates():
    #     st.write(f"- {template_name}")
    #
    # if st.button("Validate Current Protocol"):
    #     if st.session_state.get("protocol_text"):
    #         results = content_manager.validate_protocol(st.session_state.protocol_text)
    #         st.write(results)
