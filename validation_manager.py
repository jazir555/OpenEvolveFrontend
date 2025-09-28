"""
Validation Manager for OpenEvolve - Protocol validation and compliance checking
This file manages protocol validation, compliance checks, and validation-related features
File size: ~600 lines (under the 2000 line limit)
"""

import streamlit as st
from typing import Dict, List
import re
from session_utils import VALIDATION_RULES


class ValidationManager:
    """
    Manages protocol validation, compliance checks, and validation-related features
    """

    def __init__(self):
        self.validation_rules = VALIDATION_RULES

    def add_validation_rule(self, rule_name: str, rule_config: Dict) -> bool:
        """
        Add a new validation rule.

        Args:
            rule_name (str): Name of the rule
            rule_config (Dict): Configuration for the rule

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            self.validation_rules[rule_name] = rule_config
            return True
        except Exception as e:
            st.error(f"Error adding validation rule: {e}")
            return False

    def update_validation_rule(self, rule_name: str, rule_config: Dict) -> bool:
        """
        Update an existing validation rule.

        Args:
            rule_name (str): Name of the rule to update
            rule_config (Dict): New configuration for the rule

        Returns:
            bool: True if successful, False otherwise
        """
        if rule_name in self.validation_rules:
            try:
                self.validation_rules[rule_name] = rule_config
                return True
            except Exception as e:
                st.error(f"Error updating validation rule: {e}")
                return False
        else:
            st.error(f"Validation rule '{rule_name}' does not exist")
            return False

    def remove_validation_rule(self, rule_name: str) -> bool:
        """
        Remove a validation rule.

        Args:
            rule_name (str): Name of the rule to remove

        Returns:
            bool: True if successful, False otherwise
        """
        if rule_name in self.validation_rules:
            try:
                del self.validation_rules[rule_name]
                return True
            except Exception as e:
                st.error(f"Error removing validation rule: {e}")
                return False
        else:
            st.error(f"Validation rule '{rule_name}' does not exist")
            return False

    def list_validation_rules(self) -> List[str]:
        """
        List all available validation rules.

        Returns:
            List[str]: List of rule names
        """
        return list(self.validation_rules.keys())

    def get_validation_rule(self, rule_name: str) -> Dict:
        """
        Get details for a specific validation rule.

        Args:
            rule_name (str): Name of the rule

        Returns:
            Dict: Rule details or empty dict if not found
        """
        return self.validation_rules.get(rule_name, {})

    def validate_content_against_custom_rules(
        self, content: str, rule_names: List[str]
    ) -> Dict:
        """
        Validate content against a list of custom rules.

        Args:
            content (str): Content to validate
            rule_names (List[str]): List of rule names to apply

        Returns:
            Dict: Validation results
        """
        results = {
            "content_length": len(content),
            "validations": {},
            "overall_result": True,
            "error_count": 0,
            "warning_count": 0,
            "suggestion_count": 0,
        }

        for rule_name in rule_names:
            if rule_name in self.validation_rules:
                rule = self.validation_rules[rule_name]
                validation_result = self._apply_single_rule(content, rule, rule_name)
                results["validations"][rule_name] = validation_result

                if not validation_result["valid"]:
                    results["overall_result"] = False
                    results["error_count"] += len(validation_result["errors"])
                results["warning_count"] += len(validation_result["warnings"])
                results["suggestion_count"] += len(validation_result["suggestions"])

        return results

    def _apply_single_rule(self, content: str, rule: Dict, rule_name: str) -> Dict:
        """
        Apply a single validation rule to content.

        Args:
            content (str): Content to validate
            rule (Dict): Rule configuration
            rule_name (str): Name of the rule

        Returns:
            Dict: Validation result for this rule
        """
        errors = []
        warnings = []
        suggestions = []

        # Check length constraints
        if "max_length" in rule and len(content) > rule["max_length"]:
            errors.append(
                f"Content exceeds maximum length of {rule['max_length']} characters"
            )

        if "min_length" in rule and len(content) < rule["min_length"]:
            errors.append(
                f"Content is below minimum length of {rule['min_length']} characters"
            )

        # Check required sections
        if "required_sections" in rule:
            missing_sections = []
            for section in rule["required_sections"]:
                if section.lower() not in content.lower():
                    missing_sections.append(section)
            if missing_sections:
                errors.append(
                    f"Missing required sections: {', '.join(missing_sections)}"
                )

        # Check required keywords
        if "required_keywords" in rule:
            missing_keywords = []
            for keyword in rule["required_keywords"]:
                if keyword.lower() not in content.lower():
                    missing_keywords.append(keyword)
            if missing_keywords:
                warnings.append(
                    f"Consider adding these keywords: {', '.join(missing_keywords)}"
                )

        # Check forbidden patterns
        if "forbidden_patterns" in rule:
            for pattern in rule["forbidden_patterns"]:
                matches = re.findall(pattern, content)
                if matches:
                    errors.append(f"Forbidden pattern found: {matches[0][:50]}...")

        # Apply custom validation function if provided
        if "custom_validator" in rule and callable(rule["custom_validator"]):
            custom_result = rule["custom_validator"](content)
            errors.extend(custom_result.get("errors", []))
            warnings.extend(custom_result.get("warnings", []))
            suggestions.extend(custom_result.get("suggestions", []))

        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
            "suggestions": suggestions,
            "rule_name": rule_name,
            "rule_config": rule,
        }

    def run_compliance_check(
        self, content: str, compliance_framework: str = "generic"
    ) -> Dict:
        """
        Run a compliance check against a specific framework.

        Args:
            content (str): Content to check
            compliance_framework (str): Compliance framework to use

        Returns:
            Dict: Compliance check results
        """
        # For now, this uses the same validation logic as the broader validation
        # In a full implementation, this would connect to compliance databases
        return self._apply_single_rule(
            content,
            self.validation_rules.get(
                compliance_framework, self.validation_rules.get("generic", {})
            ),
            compliance_framework,
        )


# Initialize validation manager on import
validation_manager = ValidationManager()
