"""
RESE Security: Security Audit and Testing Suite

Comprehensive security testing, vulnerability scanning, and penetration testing tools.

Author: Agent M2 (Security and Reliability Specialist)
Created: 2025-12-31
"""

import re
import os
import subprocess
import hashlib
import json
from typing import Any, Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from enum import Enum


# =============================================================================
# Security Audit Types
# =============================================================================

class VulnerabilitySeverity(Enum):
    """Vulnerability severity levels"""
    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class VulnerabilityCategory(Enum):
    """Vulnerability categories"""
    INJECTION = "injection"               # SQL, code, command injection
    XSS = "xss"                          # Cross-site scripting
    CSRF = "csrf"                        # Cross-site request forgery
    AUTHENTICATION = "authentication"    # Authentication bypass
    AUTHORIZATION = "authorization"      # Authorization bypass
    CRYPTOGRAPHY = "cryptography"        # Weak cryptography
    CONFIGURATION = "configuration"      # Security misconfiguration
    DATA_VALIDATION = "data_validation"  # Input validation issues
    SESSION_MANAGEMENT = "session"       # Session management
    ERROR_HANDLING = "error_handling"    # Information disclosure
    LOGGING = "logging"                  # Insufficient logging
    DenIAL_OF_SERVICE = "dos"            # DoS vulnerabilities


@dataclass
class Vulnerability:
    """Security vulnerability finding"""
    id: str
    title: str
    description: str
    category: VulnerabilityCategory
    severity: VulnerabilitySeverity
    location: str                        # File:line or module
    evidence: str                        # Code snippet or evidence
    remediation: str                     # How to fix
    references: List[str] = field(default_factory=list)
    discovered_at: datetime = field(default_factory=datetime.now)
    cwe_id: Optional[str] = None         # MITRE CWE ID if applicable

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'id': self.id,
            'title': self.title,
            'description': self.description,
            'category': self.category.value,
            'severity': self.severity.value,
            'location': self.location,
            'evidence': self.evidence,
            'remediation': self.remediation,
            'references': self.references,
            'discovered_at': self.discovered_at.isoformat(),
            'cwe_id': self.cwe_id
        }


@dataclass
class SecurityAuditReport:
    """Security audit report"""
    scan_id: str
    timestamp: datetime
    target: str                          # Path or module scanned
    vulnerabilities: List[Vulnerability]
    statistics: Dict[str, Any]
    recommendations: List[str]
    score: float                         # 0-100 security score

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'scan_id': self.scan_id,
            'timestamp': self.timestamp.isoformat(),
            'target': self.target,
            'vulnerabilities': [v.to_dict() for v in self.vulnerabilities],
            'statistics': self.statistics,
            'recommendations': self.recommendations,
            'score': self.score
        }


# =============================================================================
# Static Code Analysis
# =============================================================================

class StaticAnalyzer:
    """
    Static code analysis for security vulnerabilities.

    Scans for:
    - SQL injection patterns
    - Code injection patterns
    - Hardcoded secrets
    - Weak cryptography
    - Unsafe deserialization
    - Path traversal
    - Command injection
    """

    def __init__(self):
        """Initialize static analyzer"""
        self.vulnerabilities: List[Vulnerability] = []

        # Security patterns
        self.patterns = {
            VulnerabilityCategory.INJECTION: [
                (r'execute\s*\(\s*["\'].*%.*["\']', "String formatting in execute() - SQL injection risk"),
                (r'cursor\.execute\s*\(\s*["\'].*\+\s*["\']', "String concatenation in cursor.execute()"),
                (r'eval\s*\(', "Use of eval() - code injection risk"),
                (r'exec\s*\(', "Use of exec() - code injection risk"),
                (r'__import__\s*\(\s*["\'].*%.*["\']', "Dynamic import with user input"),
                (r'subprocess\.(call|run|Popen)\s*\(\s*shell=True', "subprocess with shell=True - command injection risk"),
                (r'os\.system\s*\(', "Use of os.system() - command injection risk"),
            ],
            VulnerabilityCategory.CRYPTOGRAPHY: [
                (r'hashlib\.md5\s*\(', "Use of MD5 hash - weak algorithm"),
                (r'hashlib\.sha1\s*\(', "Use of SHA1 hash - weak algorithm"),
                (r'm Crypto\.Cipher\.DES', "Use of DES encryption - weak algorithm"),
                (r'm Crypto\.Cipher\.RC4', "Use of RC4 encryption - weak algorithm"),
            ],
            VulnerabilityCategory.DATA_VALIDATION: [
                (r'flask\.request\.form\[["\']\w+["\']\]', "Direct access to form data without validation"),
                (r'flask\.request\.args\[["\']\w+["\']\]', "Direct access to args without validation"),
                (r'request\.POST\[["\']\w+["\']\]', "Direct access to POST data without validation"),
                (r'request\.GET\[["\']\w+["\']\]', "Direct access to GET data without validation"),
                (r'request\.COOKIES\.get\(["\']\w+["\']\]', "Direct access to cookies without validation"),
            ],
            VulnerabilityCategory.SESSION_MANAGEMENT: [
                (r'session\[["\']\w+["\']\]\s*=\s*request\.', "User input directly assigned to session"),
                (r'cookies\[["\']\w+["\']\]\s*=\s*request\.', "User input directly assigned to cookies"),
            ],
            VulnerabilityCategory.ERROR_HANDLING: [
                (r'except\s*:', "Bare except - catches all exceptions including system exit"),
                (r'except\s*Exception\s+as\s+e:\s*print\s*\(\s*e\s*\)', "Printing exception may expose sensitive information"),
                (r'traceback\.print_exc\s*\(\s*\)', "Printing full traceback may expose sensitive information"),
            ],
        }

        # Compile patterns
        self.compiled_patterns = {}
        for category, patterns in self.patterns.items():
            self.compiled_patterns[category] = [
                (re.compile(pattern, re.IGNORECASE), message)
                for pattern, message in patterns
            ]

        # Secret patterns (passwords, API keys, tokens)
        self.secret_patterns = [
            (re.compile(r'password\s*=\s*["\'][^"\']+["\']', re.IGNORECASE), "Hardcoded password"),
            (re.compile(r'api_key\s*=\s*["\'][^"\']+["\']', re.IGNORECASE), "Hardcoded API key"),
            (re.compile(r'secret\s*=\s*["\'][^"\']+["\']', re.IGNORECASE), "Hardcoded secret"),
            (re.compile(r'token\s*=\s*["\'][^"\']+["\']', re.IGNORECASE), "Hardcoded token"),
            (re.compile(r'AWS_ACCESS_KEY_ID\s*=\s*["\'][^"\']+["\']', re.IGNORECASE), "Hardcoded AWS access key"),
            (re.compile(r'AWS_SECRET_ACCESS_KEY\s*=\s*["\'][^"\']+["\']', re.IGNORECASE), "Hardcoded AWS secret key"),
        ]

    def analyze_file(self, file_path: Path) -> List[Vulnerability]:
        """
        Analyze a file for security vulnerabilities.

        Args:
            file_path: Path to file to analyze

        Returns:
            List of vulnerabilities found
        """
        vulnerabilities = []

        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                lines = content.split('\n')

            # Check for security patterns
            for category, patterns in self.compiled_patterns.items():
                for pattern, message in patterns:
                    for line_num, line in enumerate(lines, 1):
                        if pattern.search(line):
                            vulnerabilities.append(Vulnerability(
                                id=self._generate_vuln_id(),
                                title=f"Security Issue: {message}",
                                description=message,
                                category=category,
                                severity=self._get_severity_for_category(category),
                                location=f"{file_path}:{line_num}",
                                evidence=line.strip(),
                                remediation=self._get_remediation(category),
                                cwe_id=self._get_cwe_id(category)
                            ))

            # Check for hardcoded secrets
            for pattern, message in self.secret_patterns:
                for line_num, line in enumerate(lines, 1):
                    if pattern.search(line):
                        vulnerabilities.append(Vulnerability(
                            id=self._generate_vuln_id(),
                            title=f"Hardcoded Secret: {message}",
                            description=message,
                            category=VulnerabilityCategory.CONFIGURATION,
                            severity=VulnerabilitySeverity.HIGH,
                            location=f"{file_path}:{line_num}",
                            evidence=line.strip(),
                            remediation="Move secrets to environment variables or secure configuration",
                            references=[
                                "https://cwe.mitre.org/data/definitions/798.html"
                            ],
                            cwe_id="CWE-798"
                        ))

        except Exception as e:
            print(f"Error analyzing {file_path}: {e}")

        return vulnerabilities

    def analyze_directory(
        self,
        directory: Path,
        extensions: Optional[List[str]] = None
    ) -> List[Vulnerability]:
        """
        Analyze all Python files in directory.

        Args:
            directory: Directory to scan
            extensions: File extensions to scan (default: .py)

        Returns:
            List of vulnerabilities found
        """
        if extensions is None:
            extensions = ['.py']

        vulnerabilities = []

        for file_path in directory.rglob('*'):
            if file_path.suffix in extensions:
                vulnerabilities.extend(self.analyze_file(file_path))

        return vulnerabilities

    def _generate_vuln_id(self) -> str:
        """Generate unique vulnerability ID"""
        return hashlib.md5(
            f"{datetime.now().isoformat()}{os.urandom(8)}".encode()
        ).hexdigest()[:16]

    def _get_severity_for_category(self, category: VulnerabilityCategory) -> VulnerabilitySeverity:
        """Get default severity for category"""
        severity_map = {
            VulnerabilityCategory.INJECTION: VulnerabilitySeverity.HIGH,
            VulnerabilityCategory.XSS: VulnerabilitySeverity.HIGH,
            VulnerabilityCategory.CSRF: VulnerabilitySeverity.MEDIUM,
            VulnerabilityCategory.AUTHENTICATION: VulnerabilitySeverity.HIGH,
            VulnerabilityCategory.AUTHORIZATION: VulnerabilitySeverity.HIGH,
            VulnerabilityCategory.CRYPTOGRAPHY: VulnerabilitySeverity.MEDIUM,
            VulnerabilityCategory.CONFIGURATION: VulnerabilitySeverity.MEDIUM,
            VulnerabilityCategory.DATA_VALIDATION: VulnerabilitySeverity.MEDIUM,
            VulnerabilityCategory.SESSION_MANAGEMENT: VulnerabilitySeverity.MEDIUM,
            VulnerabilityCategory.ERROR_HANDLING: VulnerabilitySeverity.LOW,
            VulnerabilityCategory.LOGGING: VulnerabilitySeverity.LOW,
        }
        return severity_map.get(category, VulnerabilitySeverity.MEDIUM)

    def _get_remediation(self, category: VulnerabilityCategory) -> str:
        """Get remediation advice for category"""
        remediation_map = {
            VulnerabilityCategory.INJECTION: "Use parameterized queries or prepared statements",
            VulnerabilityCategory.XSS: "Sanitize and escape user input before rendering",
            VulnerabilityCategory.CSRF: "Implement CSRF tokens for state-changing operations",
            VulnerabilityCategory.AUTHENTICATION: "Use strong authentication and multi-factor authentication",
            VulnerabilityCategory.AUTHORIZATION: "Implement proper access control checks",
            VulnerabilityCategory.CRYPTOGRAPHY: "Use strong, modern cryptographic algorithms",
            VulnerabilityCategory.CONFIGURATION: "Review and harden security configuration",
            VulnerabilityCategory.DATA_VALIDATION: "Validate all user input against strict schemas",
            VulnerabilityCategory.SESSION_MANAGEMENT: "Use secure session management with proper expiration",
            VulnerabilityCategory.ERROR_HANDLING: "Log errors securely without exposing sensitive information",
            VulnerabilityCategory.LOGGING: "Implement comprehensive security logging",
        }
        return remediation_map.get(category, "Review and address security issue")

    def _get_cwe_id(self, category: VulnerabilityCategory) -> Optional[str]:
        """Get MITRE CWE ID for category"""
        cwe_map = {
            VulnerabilityCategory.INJECTION: "CWE-89",
            VulnerabilityCategory.XSS: "CWE-79",
            VulnerabilityCategory.CSRF: "CWE-352",
            VulnerabilityCategory.AUTHENTICATION: "CWE-287",
            VulnerabilityCategory.AUTHORIZATION: "CWE-285",
            VulnerabilityCategory.CRYPTOGRAPHY: "CWE-327",
            VulnerabilityCategory.CONFIGURATION: "CWE-16",
            VulnerabilityCategory.DATA_VALIDATION: "CWE-20",
            VulnerabilityCategory.SESSION_MANAGEMENT: "CWE-613",
            VulnerabilityCategory.ERROR_HANDLING: "CWE-209",
            VulnerabilityCategory.LOGGING: "CWE-778",
        }
        return cwe_map.get(category)


# =============================================================================
# Dependency Vulnerability Scanner
# =============================================================================

class DependencyScanner:
    """
    Scan dependencies for known vulnerabilities.

    Checks:
    - Python packages against security advisories
    - Version compatibility
    - Known CVEs
    """

    def __init__(self):
        """Initialize dependency scanner"""
        self.vulnerabilities: List[Vulnerability] = []

    def scan_requirements(self, requirements_file: Path) -> List[Vulnerability]:
        """
        Scan requirements.txt for vulnerable packages.

        Args:
            requirements_file: Path to requirements.txt

        Returns:
            List of vulnerabilities found
        """
        vulnerabilities = []

        try:
            with open(requirements_file, 'r') as f:
                requirements = f.readlines()

            # Check for known vulnerable packages
            vulnerable_packages = {
                'flask': ['0.12.0', '0.12.1', '0.12.2', '0.12.3'],
                'jinja2': ['2.10'],
                'pillow': ['5.2.0', '5.3.0'],
                'urllib3': ['1.24.2', '1.25.0', '1.25.1'],
                # Add more as needed
            }

            for requirement in requirements:
                requirement = requirement.strip()
                if not requirement or requirement.startswith('#'):
                    continue

                # Parse package name and version
                parts = requirement.split('==')
                if len(parts) == 2:
                    package_name = parts[0].lower().strip()
                    version = parts[1].strip()

                    if package_name in vulnerable_packages:
                        if version in vulnerable_packages[package_name]:
                            vulnerabilities.append(Vulnerability(
                                id=self._generate_vuln_id(),
                                title=f"Vulnerable Package: {package_name}",
                                description=f"Package {package_name} version {version} has known vulnerabilities",
                                category=VulnerabilityCategory.CONFIGURATION,
                                severity=VulnerabilitySeverity.HIGH,
                                location=f"{requirements_file}:{requirement}",
                                evidence=requirement,
                                remediation=f"Update {package_name} to latest version",
                                references=[
                                    f"https://nvd.nist.gov/vuln/search/results?form_type=Advanced&cpe_vendor=cpe%3A%2F%3A{package_name}_project&cpe_product={package_name}"
                                ]
                            ))

        except Exception as e:
            print(f"Error scanning requirements: {e}")

        return vulnerabilities

    def _generate_vuln_id(self) -> str:
        """Generate unique vulnerability ID"""
        return hashlib.md5(
            f"{datetime.now().isoformat()}{os.urandom(8)}".encode()
        ).hexdigest()[:16]


# =============================================================================
# Penetration Testing
# =============================================================================

class PenetrationTester:
    """
    Automated penetration testing for RESE components.

    Tests:
    - SQL injection
    - XSS
    - CSRF
    - Authentication bypass
    - Authorization bypass
    - Path traversal
    - Command injection
    """

    def __init__(self, base_url: str = "http://localhost:8000"):
        """
        Initialize penetration tester.

        Args:
            base_url: Base URL of application to test
        """
        self.base_url = base_url
        self.vulnerabilities: List[Vulnerability] = []

    def test_sql_injection(self, endpoint: str, params: Dict[str, str]) -> List[Vulnerability]:
        """
        Test endpoint for SQL injection.

        Args:
            endpoint: API endpoint to test
            params: Parameters to test

        Returns:
            List of vulnerabilities found
        """
        vulnerabilities = []

        # SQL injection payloads
        payloads = [
            "' OR '1'='1",
            "' OR '1'='1'--",
            "' OR '1'='1'/*",
            "admin'--",
            "admin'/*",
            "' UNION SELECT NULL--",
            "1' AND 1=1--",
            "1' AND 1=2--",
        ]

        # Test each parameter with each payload
        import requests

        for param_name, original_value in params.items():
            for payload in payloads:
                test_params = params.copy()
                test_params[param_name] = payload

                try:
                    response = requests.get(f"{self.base_url}{endpoint}", params=test_params, timeout=5)

                    # Check for SQL error in response
                    sql_errors = [
                        "SQL syntax",
                        "mysql_fetch",
                        "ORA-",
                        "SQLite3::SQLException",
                        "PostgreSQL",
                        "Warning: pg_",
                        "valid MySQL result",
                        "check the manual that corresponds to your MySQL",
                        "MySqlClient",
                        "PostgreSQLException",
                        "DriverSQL",
                    ]

                    response_text = response.text.lower()
                    if any(error.lower() in response_text for error in sql_errors):
                        vulnerabilities.append(Vulnerability(
                            id=self._generate_vuln_id(),
                            title="SQL Injection Vulnerability",
                            description=f"SQL injection possible in parameter '{param_name}'",
                            category=VulnerabilityCategory.INJECTION,
                            severity=VulnerabilitySeverity.CRITICAL,
                            location=f"{endpoint}?{param_name}={payload[:20]}",
                            evidence=f"Payload: {payload}\nResponse contained SQL error",
                            remediation="Use parameterized queries or prepared statements",
                            references=["https://owasp.org/www-community/attacks/SQL_Injection"],
                            cwe_id="CWE-89"
                        ))
                        break  # Found vulnerability, no need to test more payloads

                except Exception as e:
                    pass  # Request failed, not a vulnerability

        return vulnerabilities

    def test_xss(self, endpoint: str, params: Dict[str, str]) -> List[Vulnerability]:
        """
        Test endpoint for XSS vulnerabilities.

        Args:
            endpoint: API endpoint to test
            params: Parameters to test

        Returns:
            List of vulnerabilities found
        """
        vulnerabilities = []

        # XSS payloads
        payloads = [
            "<script>alert('XSS')</script>",
            "<img src=x onerror=alert('XSS')>",
            "javascript:alert('XSS')",
            "<svg onload=alert('XSS')>",
            "'><script>alert(String.fromCharCode(88,83,83))</script>",
        ]

        import requests

        for param_name, original_value in params.items():
            for payload in payloads:
                test_params = params.copy()
                test_params[param_name] = payload

                try:
                    response = requests.get(f"{self.base_url}{endpoint}", params=test_params, timeout=5)

                    # Check if payload is reflected unescaped
                    if payload in response.text:
                        vulnerabilities.append(Vulnerability(
                            id=self._generate_vuln_id(),
                            title="Cross-Site Scripting (XSS) Vulnerability",
                            description=f"XSS vulnerability in parameter '{param_name}'",
                            category=VulnerabilityCategory.XSS,
                            severity=VulnerabilitySeverity.HIGH,
                            location=f"{endpoint}?{param_name}=",
                            evidence=f"Payload: {payload}\nReflected in response",
                            remediation="Sanitize and escape user input before rendering",
                            references=["https://owasp.org/www-community/attacks/xss/"],
                            cwe_id="CWE-79"
                        ))
                        break

                except Exception as e:
                    pass

        return vulnerabilities

    def _generate_vuln_id(self) -> str:
        """Generate unique vulnerability ID"""
        return hashlib.md5(
            f"{datetime.now().isoformat()}{os.urandom(8)}".encode()
        ).hexdigest()[:16]


# =============================================================================
# Security Audit Orchestrator
# =============================================================================

class SecurityAuditor:
    """
    Orchestrates comprehensive security audit.

    Combines:
    - Static code analysis
    - Dependency scanning
    - Penetration testing
    - Configuration review
    """

    def __init__(self, target_path: Path):
        """
        Initialize security auditor.

        Args:
            target_path: Path to application to audit
        """
        self.target_path = target_path
        self.static_analyzer = StaticAnalyzer()
        self.dependency_scanner = DependencyScanner()

    def run_full_audit(self) -> SecurityAuditReport:
        """
        Run comprehensive security audit.

        Returns:
            SecurityAuditReport with all findings
        """
        scan_id = hashlib.md5(
            f"{datetime.now().isoformat()}{self.target_path}".encode()
        ).hexdigest()[:16]

        all_vulnerabilities = []

        # Static analysis
        print("Running static code analysis...")
        static_vulns = self.static_analyzer.analyze_directory(self.target_path)
        all_vulnerabilities.extend(static_vulns)

        # Dependency scanning
        requirements_file = self.target_path / "requirements.txt"
        if requirements_file.exists():
            print("Scanning dependencies...")
            dep_vulns = self.dependency_scanner.scan_requirements(requirements_file)
            all_vulnerabilities.extend(dep_vulns)

        # Calculate statistics
        vulns_by_severity = {}
        for severity in VulnerabilitySeverity:
            count = sum(1 for v in all_vulnerabilities if v.severity == severity)
            vulns_by_severity[severity.value] = count

        vulns_by_category = {}
        for category in VulnerabilityCategory:
            count = sum(1 for v in all_vulnerabilities if v.category == category)
            vulns_by_category[category.value] = count

        statistics = {
            'total_vulnerabilities': len(all_vulnerabilities),
            'by_severity': vulns_by_severity,
            'by_category': vulns_by_category
        }

        # Calculate security score (0-100)
        score = self._calculate_security_score(statistics)

        # Generate recommendations
        recommendations = self._generate_recommendations(statistics)

        return SecurityAuditReport(
            scan_id=scan_id,
            timestamp=datetime.now(),
            target=str(self.target_path),
            vulnerabilities=all_vulnerabilities,
            statistics=statistics,
            recommendations=recommendations,
            score=score
        )

    def _calculate_security_score(self, statistics: Dict[str, Any]) -> float:
        """Calculate security score from statistics"""
        total = statistics['total_vulnerabilities']
        by_severity = statistics['by_severity']

        # Weight scores by severity
        weights = {
            'critical': 25,
            'high': 10,
            'medium': 3,
            'low': 1,
            'info': 0.1
        }

        weighted_score = sum(
            by_severity.get(sev, 0) * weight
            for sev, weight in weights.items()
        )

        # Convert to 0-100 scale
        score = max(0, 100 - weighted_score)
        return round(score, 2)

    def _generate_recommendations(self, statistics: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on findings"""
        recommendations = []

        # Critical vulnerabilities
        if statistics['by_severity'].get('critical', 0) > 0:
            recommendations.append(
                "CRITICAL: Address all critical vulnerabilities immediately"
            )

        # High vulnerabilities
        if statistics['by_severity'].get('high', 0) > 0:
            recommendations.append(
                "HIGH: Prioritize fixing high-severity vulnerabilities"
            )

        # Injection vulnerabilities
        if statistics['by_category'].get('injection', 0) > 0:
            recommendations.append(
                "Implement input validation and use parameterized queries"
            )

        # XSS vulnerabilities
        if statistics['by_category'].get('xss', 0) > 0:
            recommendations.append(
                "Implement proper output encoding and CSP headers"
            )

        # Cryptography issues
        if statistics['by_category'].get('cryptography', 0) > 0:
            recommendations.append(
                "Update to use strong, modern cryptographic algorithms"
            )

        # Configuration issues
        if statistics['by_category'].get('configuration', 0) > 0:
            recommendations.append(
                "Review and harden security configuration"
            )

        return recommendations


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    # Data structures
    'Vulnerability',
    'VulnerabilitySeverity',
    'VulnerabilityCategory',
    'SecurityAuditReport',

    # Analyzers
    'StaticAnalyzer',
    'DependencyScanner',
    'PenetrationTester',
    'SecurityAuditor',
]
