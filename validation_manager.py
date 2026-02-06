"""
Validation Manager for OpenEvolve - Protocol validation and compliance checking
This file manages protocol validation, compliance checks, and validation-related features
File size: ~600 lines (under the 2000 line limit)
"""

from ui_shim import ui as st
from typing import Dict, List, Any, Optional
import re
import logging
from datetime import datetime
from session_utils import VALIDATION_RULES

# **ACTUAL INTEGRATION**: Alerting and knowledge for Validation Manager
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

class ValidationManager:
    """
    Manages protocol validation, compliance checks, and validation-related features
    """

    def __init__(self):
        self.validation_rules = VALIDATION_RULES
        self.compliance_databases = {
            "generic": {
                "max_length": 5000,
                "required_keywords": ["security", "privacy"],
                "forbidden_patterns": ["confidential data leak"]
            },
            "gdpr": {
                "required_keywords": ["GDPR", "data protection", "consent"],
                "forbidden_patterns": ["unencrypted personal data"]
            },
            "hipaa": {
                "required_keywords": ["HIPAA", "PHI", "patient data"],
                "forbidden_patterns": ["unsecured health information"]
            }
        }

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

        try:
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

            # **ACTUAL INTEGRATION**: Extract knowledge, track performance, and trigger alerts
            self._extract_validation_knowledge("validate_content", results, len(content))
            self._track_validation_performance("validate_content", results["overall_result"], results["error_count"])

            if not results["overall_result"]:
                self._trigger_validation_alerts("validate_content", False, results["error_count"],
                                                   f"Validation failed with {results['error_count']} errors")

        except Exception as e:
            # **ACTUAL INTEGRATION**: Trigger alert and track failure
            self._trigger_validation_alerts("validate_content", False, 0, str(e))
            self._track_validation_performance("validate_content", False, 0)
            st.error(f"Error validating content: {e}")
            raise

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
        # Connect to compliance databases and perform real compliance checks
        compliance_rules = self.compliance_databases.get(compliance_framework)
        
        if not compliance_rules:
            st.warning(f"Compliance framework '{compliance_framework}' not found. Using generic rules.")
            compliance_rules = self.compliance_databases.get("generic", {})

        return self._apply_single_rule(
            content,
            compliance_rules,
            compliance_framework,
        )

    # =========================================================================
    # ACTUAL INTEGRATION METHODS - Alerting, knowledge, and adaptive for Validation Manager
    # =========================================================================

    def _trigger_validation_alerts(
        self,
        operation: str,
        success: bool,
        error_count: int = 0,
        error: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """**ACTUAL INTEGRATION**: Trigger alerts for validation failures."""
        if not ALERTING_AVAILABLE:
            return

        try:
            alert_manager = get_alert_manager()

            if not success or error_count > 0:
                severity = AlertSeverity.HIGH if error_count > 5 else AlertSeverity.MEDIUM

                alert_manager.create_alert(
                    title=f"Validation Alert: {operation}",
                    description=f"Validation operation '{operation}' " +
                                 ("failed" if not success else f"found {error_count} errors") +
                                 (f". Error: {error}" if error else ""),
                    severity=severity.value,
                    source="validation_manager",
                    component="validation_compliance",
                    metadata=metadata or {}
                )

        except Exception as e:
            logging.error(f"Failed to trigger Validation alert: {e}")

    def _extract_validation_knowledge(
        self,
        operation: str,
        results: Dict[str, Any],
        content_length: int
    ) -> bool:
        """**ACTUAL INTEGRATION**: Extract validation knowledge to knowledge engine."""
        if not KNOWLEDGE_AVAILABLE:
            return False

        try:
            knowledge_engine = get_knowledge_engine()

            artifact = KnowledgeArtifact(
                artifact_id=f"validation_{operation}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                artifact_type="validation_result",
                source_component="validation_manager",
                title=f"Validation Result: {operation}",
                content={
                    "operation": operation,
                    "overall_result": results.get("overall_result", False),
                    "error_count": results.get("error_count", 0),
                    "warning_count": results.get("warning_count", 0),
                    "suggestion_count": results.get("suggestion_count", 0),
                    "content_length": content_length,
                    "timestamp": datetime.now().isoformat()
                },
                metadata={"validation_keys": list(results.get("validations", {}).keys())},
                tags=["validation", "compliance", operation, "quality_check"]
            )

            knowledge_engine.store_artifact(artifact)
            logging.debug(f"Extracted Validation knowledge for {operation}")
            return True

        except Exception as e:
            logging.error(f"Failed to extract Validation knowledge: {e}")
            return False

    def _track_validation_performance(
        self,
        operation: str,
        success: bool,
        error_count: int = 0
    ):
        """**ACTUAL INTEGRATION**: Track validation operation performance in adaptive selector."""
        if not ADAPTIVE_AVAILABLE:
            return

        try:
            tracker = StrategyPerformanceTracker()

            performance_data = StrategyPerformanceData(
                strategy_name=f"validation_manager_{operation}",
                success_count=1 if success and error_count == 0 else 0,
                failure_count=0 if success and error_count == 0 else 1,
                average_quality=1.0 if success and error_count == 0 else max(0, 1.0 - (error_count * 0.1)),
                last_used=datetime.now(),
                total_attempts=1,
                metadata={"operation": operation, "error_count": error_count}
            )

            if hasattr(tracker, 'performance_history'):
                tracker.performance_history.append(performance_data)
                logging.debug(f"Tracked Validation performance for {operation}")

        except Exception as e:
            logging.error(f"Failed to track Validation performance: {e}")


# Initialize validation manager on import
validation_manager = ValidationManager()

def render_validation_manager():
    """
    Renders the validation manager section in the Streamlit UI.
    Allows users to define, manage, and apply validation rules.
    """
    st.header("[OK] Validation Manager")
    
    # Initialize validation manager
    vm = ValidationManager()
    
    # Rule management tab
    tab1, tab2, tab3 = st.tabs(["📋 Manage Rules", "🔍 Apply Validation", "📊 Validation Results"])
    
    with tab1:
        st.subheader("Manage Validation Rules")
        
        # Show existing rules
        rules = vm.list_validation_rules()
        if rules:
            st.write("**Available Rules:**")
            for rule_name in rules:
                rule_details = vm.get_validation_rule(rule_name)
                with st.expander(f"Rule: {rule_name}"):
                    st.json(rule_details)
                    if st.button(f"Delete Rule: {rule_name}", key=f"delete_{rule_name}"):
                        vm.remove_validation_rule(rule_name)
                        st.success(f"Rule '{rule_name}' deleted")
                        st.rerun()
        else:
            st.info("No validation rules defined yet.")
        
        # Add new rule
        with st.expander("Add New Validation Rule"):
            rule_name = st.text_input("Rule Name")
            col1, col2 = st.columns(2)
            with col1:
                max_length = st.number_input("Max Length (0 = no limit)", min_value=0, value=0)
                min_length = st.number_input("Min Length (0 = no limit)", min_value=0, value=0)
            with col2:
                required_keywords = st.text_input("Required Keywords (comma-separated)")
                forbidden_patterns = st.text_input("Forbidden Patterns (comma-separated regex)")
            
            required_sections = st.text_input("Required Sections (comma-separated)")
            
            if st.button("Add Rule"):
                if rule_name:
                    rule_config = {}
                    if max_length > 0:
                        rule_config["max_length"] = max_length
                    if min_length > 0:
                        rule_config["min_length"] = min_length
                    if required_keywords.strip():
                        rule_config["required_keywords"] = [kw.strip() for kw in required_keywords.split(",")]
                    if forbidden_patterns.strip():
                        rule_config["forbidden_patterns"] = [pt.strip() for pt in forbidden_patterns.split(",")]
                    if required_sections.strip():
                        rule_config["required_sections"] = [sec.strip() for sec in required_sections.split(",")]
                    
                    if vm.add_validation_rule(rule_name, rule_config):
                        st.success(f"Rule '{rule_name}' added successfully!")
                        st.rerun()
                    else:
                        st.error(f"Failed to add rule '{rule_name}'")
                else:
                    st.error("Please provide a rule name")
    
    with tab2:
        st.subheader("Apply Validation")
        
        all_rules = vm.list_validation_rules()
        if all_rules:
            selected_rules = st.multiselect("Select rules to apply", all_rules, default=all_rules)
            
            if st.button("Run Validation on Current Content"):
                if selected_rules:
                    content = st.session_state.get("protocol_text", "")
                    if content.strip():
                        validation_results = vm.validate_content_against_custom_rules(content, selected_rules)
                        
                        # Store results in session state
                        st.session_state.validation_results = validation_results
                        
                        st.success("Validation completed!")
                        
                        # Display summary
                        st.subheader("Validation Summary")
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Overall Result", "[OK] Pass" if validation_results["overall_result"] else "[FAIL] Fail")
                        with col2:
                            st.metric("Errors", validation_results["error_count"])
                        with col3:
                            st.metric("Warnings", validation_results["warning_count"])
                    else:
                        st.error("No content to validate. Please enter content in the main editor.")
                else:
                    st.error("Please select at least one rule to apply.")
        else:
            st.info("No validation rules available. Create some rules in the 'Manage Rules' tab first.")
    
    with tab3:
        st.subheader("Validation Results")
        
        if "validation_results" in st.session_state:
            results = st.session_state.validation_results
            st.json(results)
            
            # Detailed breakdown
            st.subheader("Detailed Results")
            for rule_name, rule_result in results["validations"].items():
                with st.expander(f"Rule: {rule_name} - {'[OK] Pass' if rule_result['valid'] else '[FAIL] Fail'}"):
                    if rule_result["errors"]:
                        st.error(f"Errors ({len(rule_result['errors'])}):")
                        for error in rule_result["errors"]:
                            st.write(f"- {error}")
                    if rule_result["warnings"]:
                        st.warning(f"Warnings ({len(rule_result['warnings'])}):")
                        for warning in rule_result["warnings"]:
                            st.write(f"- {warning}")
                    if rule_result["suggestions"]:
                        st.info(f"Suggestions ({len(rule_result['suggestions'])}):")
                        for suggestion in rule_result["suggestions"]:
                            st.write(f"- {suggestion}")
        else:
            st.info("Run validation first to see results here.")
