"""
Verification Node for BubbleLabs Integration

Implements solution verification using Lean4, automated testing, and statistical validation.
"""

from typing import Dict, Any, List, Optional
from .base_node import BubbleLabsNode, NodeExecutionError


class VerificationNode(BubbleLabsNode):
    """
    Verifies solutions using multiple methods.

    Supports verification methods:
    - lean4: Formal mathematical verification
    - automated: Automated testing and validation
    - statistical: Statistical analysis and confidence intervals
    - peer_review: Cross-model peer review
    """

    # Node metadata
    DISPLAY_NAME = "Solution Verification"
    DESCRIPTION = (
        "Verify solutions using formal methods, automated testing, "
        "statistical validation, and peer review."
    )
    ICON = "verification"
    CATEGORY = "verification"
    VERSION = "1.0.0"

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)

        # Import verification engine (safe import)
        VerificationEngine = self.safe_import(
            'verification_engine.VerificationEngine',
            fallback_value=None,
            error_msg="VerificationEngine not available for VerificationNode"
        )

        if VerificationEngine:
            try:
                self.engine = VerificationEngine()
            except Exception as e:
                self.logger.warning(f"Could not instantiate VerificationEngine: {e}")
                self.engine = None
        else:
            self.engine = None

    def validate_inputs(self, inputs: Dict) -> List[str]:
        """
        Validate input parameters.

        Required:
            - solution: Dict containing solution to verify

        Optional:
            - verification_methods: List[str]
            - strictness: str (lenient, standard, strict)
            - timeout: int
        """
        errors = []

        # Check required fields
        if 'solution' not in inputs:
            errors.append("Missing required field: solution")
        elif not isinstance(inputs['solution'], dict):
            errors.append("solution must be a dictionary")

        # Validate verification_methods
        if 'verification_methods' in inputs:
            if not isinstance(inputs['verification_methods'], list):
                errors.append("verification_methods must be a list")
            else:
                valid_methods = ['lean4', 'automated', 'statistical', 'peer_review']
                for vm in inputs['verification_methods']:
                    if vm not in valid_methods:
                        errors.append(f"Invalid verification method: {vm}. Must be one of {valid_methods}")

        # Validate strictness
        if 'strictness' in inputs:
            valid_levels = ['lenient', 'standard', 'strict']
            if inputs['strictness'] not in valid_levels:
                errors.append(f"strictness must be one of: {', '.join(valid_levels)}")

        # Validate timeout
        if 'timeout' in inputs:
            if not isinstance(inputs['timeout'], int):
                errors.append("timeout must be an integer")
            elif inputs['timeout'] < 0:
                errors.append("timeout must be non-negative")

        return errors

    def execute(self, inputs: Dict, context) -> Dict[str, Any]:
        """
        Verify a solution using specified methods.

        Args:
            inputs: Must contain 'solution' and optional verification parameters
            context: Workflow state for tracking

        Returns:
            Dict containing:
                - verified: Whether solution passed verification
                - confidence: Overall verification confidence (0-1)
                - verification_reports: List of verification reports
                - issues_found: List of issues discovered
                - certification_level: Certification level achieved
        """
        if not self.engine:
            return self._verify_simple(inputs, context)

        solution = inputs['solution']
        verification_methods = inputs.get('verification_methods', self.config.get('verification_methods', [
            'automated',
            'statistical'
        ]))
        strictness = inputs.get('strictness', self.config.get('strictness', 'standard'))
        timeout = inputs.get('timeout', self.config.get('timeout', 300))

        # Update progress
        context.update_progress(10, f"Initializing verification (strictness: {strictness})")
        self.logger.info(f"Verifying solution using methods: {', '.join(verification_methods)}")

        try:
            # Run verifications
            context.update_progress(20, "Starting verification process")

            verification_result = self.engine.verify(
                solution=solution,
                methods=verification_methods,
                strictness=strictness,
                timeout=timeout,
                callback=lambda p, m: context.update_progress(20 + p * 0.7, m)
            )

            # Update progress
            context.update_progress(90, "Processing verification results")

            # Determine certification level
            certification_level = self._determine_certification(
                verification_result.overall_confidence,
                verification_result.issues,
                strictness
            )

            # Extract and format results
            result = {
                'verified': verification_result.verified,
                'confidence': verification_result.overall_confidence,
                'verification_reports': self._format_reports(verification_result.reports),
                'issues_found': self._format_issues(verification_result.issues),
                'certification_level': certification_level,
                'summary': {
                    'methods_used': verification_methods,
                    'strictness': strictness,
                    'total_issues': len(verification_result.issues),
                    'critical_issues': len([i for i in verification_result.issues if i.severity == 'critical']),
                    'warnings': len([i for i in verification_result.issues if i.severity == 'warning']),
                    'suggestions': len([i for i in verification_result.issues if i.severity == 'suggestion'])
                },
                'metadata': {
                    'verification_time': verification_result.verification_time,
                    'verifier_version': verification_result.version,
                    'timeout_used': timeout
                }
            }

            # Add artifacts to context
            context.add_artifact('verification', {
                'result': result,
                'solution_id': solution.get('id', 'unknown'),
                'methods': verification_methods
            })

            status_msg = "VERIFIED" if result['verified'] else "FAILED"
            context.update_progress(
                100,
                f"Verification {status_msg}: confidence={result['confidence']:.2f}, "
                f"certification={certification_level}, "
                f"issues={result['summary']['total_issues']}"
            )

            self.logger.info(
                f"Verification completed: {status_msg}, "
                f"confidence={result['confidence']:.2f}, "
                f"certification={certification_level}"
            )

            return result

        except Exception as e:
            self.logger.error(f"Verification failed: {str(e)}", exc_info=True)
            raise NodeExecutionError(
                node_name=self.get_display_name(),
                message=f"Verification failed: {str(e)}",
                details={
                    'solution_id': solution.get('id', 'unknown'),
                    'methods': verification_methods,
                    'strictness': strictness,
                    'exception_type': type(e).__name__
                }
            ) from e

    def _verify_simple(self, inputs: Dict, context) -> Dict[str, Any]:
        """Simple verification fallback when engine not available"""
        solution = inputs['solution']
        verification_methods = inputs.get('verification_methods', ['automated'])
        strictness = inputs.get('strictness', 'standard')

        context.update_progress(10, "Using simple verification (engine not available)")

        import time
        start_time = time.time()

        context.update_progress(30, "Performing basic checks")

        # Simple verification checks
        issues = []
        verified = True

        # Check if solution has required fields
        if not solution.get('description'):
            issues.append({
                'severity': 'warning',
                'message': 'Solution missing description',
                'location': 'solution.description'
            })
            verified = False

        if not solution.get('approach'):
            issues.append({
                'severity': 'suggestion',
                'message': 'Solution should specify approach',
                'location': 'solution.approach'
            })

        verification_time = time.time() - start_time

        # Calculate simple confidence
        base_confidence = 0.5
        if verified:
            base_confidence += 0.3
        if len(issues) == 0:
            base_confidence += 0.2

        result = {
            'verified': verified and strictness != 'strict',
            'confidence': min(base_confidence, 1.0),
            'verification_reports': [
                {
                    'method': 'basic_checks',
                    'passed': verified,
                    'confidence': base_confidence,
                    'details': 'Basic structural verification performed'
                }
            ],
            'issues_found': issues,
            'certification_level': self._determine_certification(base_confidence, [], strictness),
            'summary': {
                'methods_used': verification_methods,
                'strictness': strictness,
                'total_issues': len(issues),
                'critical_issues': 0,
                'warnings': len([i for i in issues if i['severity'] == 'warning']),
                'suggestions': len([i for i in issues if i['severity'] == 'suggestion'])
            },
            'metadata': {
                'verification_time': verification_time,
                'warning': 'Full engine not available, using basic checks'
            }
        }

        context.update_progress(100, f"Simple verification complete in {verification_time:.2f}s")
        return result

    def _determine_certification(self, confidence: float, issues: List, strictness: str) -> str:
        """Determine certification level based on results"""
        if confidence >= 0.95 and len(issues) == 0:
            return 'gold'
        elif confidence >= 0.85 and strictness != 'strict':
            return 'silver'
        elif confidence >= 0.7:
            return 'bronze'
        else:
            return 'provisional'

    def _format_reports(self, reports: List) -> List[Dict[str, Any]]:
        """Format verification reports for output"""
        formatted = []

        for report in reports:
            formatted.append({
                'method': report.method,
                'passed': report.passed,
                'confidence': report.confidence,
                'details': report.details,
                'timestamp': report.timestamp,
                'verifier_info': report.verifier_info
            })

        return formatted

    def _format_issues(self, issues: List) -> List[Dict[str, Any]]:
        """Format issues for output"""
        formatted = []

        for issue in issues:
            formatted.append({
                'severity': getattr(issue, 'severity', 'unknown'),
                'category': getattr(issue, 'category', 'general'),
                'message': getattr(issue, 'message', ''),
                'location': getattr(issue, 'location', 'unknown'),
                'suggestion': getattr(issue, 'suggestion', None)
            })

        return formatted

    def get_parameter_schema(self) -> Dict[str, Any]:
        """Get JSON schema for node parameters"""
        return {
            "type": "object",
            "title": "Verification Configuration",
            "description": "Configure solution verification parameters",
            "properties": {
                "verification_methods": {
                    "type": "array",
                    "title": "Verification Methods",
                    "description": "Methods to use for verification",
                    "items": {
                        "type": "string",
                        "enum": ["lean4", "automated", "statistical", "peer_review"]
                    },
                    "uniqueItems": True,
                    "default": ["automated", "statistical"]
                },
                "strictness": {
                    "type": "string",
                    "title": "Strictness Level",
                    "description": "How strict to be during verification",
                    "enum": ["lenient", "standard", "strict"],
                    "enumNames": [
                        "Lenient (Allow minor issues)",
                        "Standard (Require quality)",
                        "Strict (Require perfection)"
                    ],
                    "default": "standard"
                },
                "timeout": {
                    "type": "integer",
                    "title": "Timeout",
                    "description": "Maximum verification time in seconds",
                    "minimum": 0,
                    "maximum": 3600,
                    "default": 300
                },
                "require_all_methods": {
                    "type": "boolean",
                    "title": "Require All Methods",
                    "description": "All verification methods must pass for overall success",
                    "default": False
                },
                "enable_cross_validation": {
                    "type": "boolean",
                    "title": "Enable Cross-Validation",
                    "description": "Cross-validate results across multiple models",
                    "default": True
                }
            },
            "required": ["verification_methods", "strictness"]
        }
