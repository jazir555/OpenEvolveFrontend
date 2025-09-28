"""
Session Utilities for OpenEvolve - Core utilities and helper functions
This file contains utility functions and core helpers that were in the original sessionstate.py
File size: ~1500 lines (under the 2000 line limit)
"""

import streamlit as st
import threading
from typing import Any, Dict, List, Optional
from datetime import datetime
import json
import hashlib
import re
from providers import PROVIDERS


# Thread lock for safely updating shared session state from background threads.
# Use a lock to ensure thread-safe initialization of the session state lock
if "thread_lock" not in st.session_state:
    with threading.Lock():
        # Double-checked locking pattern to ensure thread safety
        if "thread_lock" not in st.session_state:
            st.session_state.thread_lock = threading.Lock()

# Real-time collaboration state management
if "collaboration_session" not in st.session_state:
    st.session_state.collaboration_session = {
        "active_users": [],
        "last_activity": datetime.now().timestamp()
        * 1000,  # Using datetime.now().timestamp() * 1000 for _now_ms()
        "chat_messages": [],
        "notifications": [],
        "shared_cursor_position": 0,
        "edit_locks": {},
    }


def calculate_protocol_complexity(protocol_text: str) -> Dict:
    words = protocol_text.split()
    word_count = len(words)
    sentences = re.split(r"[.!?]", protocol_text)
    sentence_count = len([s for s in sentences if s.strip()])
    paragraphs = protocol_text.split("\n\n")
    paragraph_count = len([p for p in paragraphs if p.strip()])
    unique_words = len(set(w.lower() for w in words))
    avg_sentence_length = word_count / max(1, sentence_count)

    # Simple complexity score: more words, longer sentences, fewer unique words = higher complexity
    complexity_score = (
        (word_count / 100) + (avg_sentence_length * 2) - (unique_words / 50)
    )
    complexity_score = max(0, min(100, complexity_score))  # Clamp between 0 and 100

    return {
        "word_count": word_count,
        "sentence_count": sentence_count,
        "paragraph_count": paragraph_count,
        "complexity_score": round(complexity_score, 2),
        "unique_words": unique_words,
        "avg_sentence_length": round(avg_sentence_length, 2),
    }


def extract_protocol_structure(protocol_text: str) -> Dict:
    has_headers = bool(re.search(r"^#+\s", protocol_text, re.MULTILINE))
    has_numbered_steps = bool(re.search(r"^\d+\.+", protocol_text, re.MULTILINE))
    has_bullet_points = bool(re.search(r"^-+\s", protocol_text, re.MULTILINE))
    has_preconditions = "preconditions" in protocol_text.lower()
    has_postconditions = "postconditions" in protocol_text.lower()
    has_error_handling = (
        "error handling" in protocol_text.lower()
        or "exception handling" in protocol_text.lower()
    )
    section_count = len(re.findall(r"^#+\s", protocol_text, re.MULTILINE)) + len(
        re.findall(r"^.*\n[=]{3,}", protocol_text, re.MULTILINE)
    )

    return {
        "has_headers": has_headers,
        "has_numbered_steps": has_numbered_steps,
        "has_bullet_points": has_bullet_points,
        "has_preconditions": has_preconditions,
        "has_postconditions": has_postconditions,
        "has_error_handling": has_error_handling,
        "section_count": section_count,
    }


def generate_protocol_recommendations(protocol_text: str) -> List[str]:
    recommendations = []
    if len(protocol_text.split()) < 50:
        recommendations.append("Consider expanding the protocol with more details.")
    if "todo" in protocol_text.lower():
        recommendations.append("Address any 'TODO' items in the protocol.")
    if not re.search(r"^\s*#+\s", protocol_text, re.MULTILINE):
        recommendations.append("Add clear section headers to improve readability.")
    if not recommendations:
        recommendations.append("Protocol looks good, no immediate recommendations.")

    return recommendations


def _clamp(x: float, min_val: float, max_val: float) -> float:
    """Clamps a value between a minimum and maximum."""
    return max(min_val, min(max_val, x))


def _rand_jitter_ms() -> float:
    """Returns a random jitter value in milliseconds."""
    import random

    return random.uniform(0.01, 0.1)


def _approx_tokens(text: str) -> int:
    """Approximates the number of tokens in a text."""
    # Rough approximation: 1 token ≈ 4 characters or 0.75 words
    if not text:
        return 0
    return max(1, len(text) // 4)


def _cost_estimate(
    prompt_tokens: int,
    completion_tokens: int,
    input_cost_per_million: float,
    output_cost_per_million: float,
) -> float:
    """Estimates the cost of a request."""
    if input_cost_per_million is None or output_cost_per_million is None:
        # Return a default cost estimate based on OpenAI pricing if costs are not provided
        return (prompt_tokens * 0.0000005) + (
            completion_tokens * 0.0000015
        )  # GPT-3.5-turbo pricing as fallback
    return (prompt_tokens / 1_000_000 * input_cost_per_million) + (
        completion_tokens / 1_000_000 * output_cost_per_million
    )


def safe_int(x: Any, default: int) -> int:
    """Safely converts a value to int, returning default if conversion fails."""
    try:
        return int(x)
    except (TypeError, ValueError):
        return default


def safe_float(x: Any, default: float) -> float:
    """Safely converts a value to float, returning default if conversion fails."""
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def _safe_list(x: Any, key: str) -> List:
    """Safely extracts a list from a dictionary, returning an empty list if the key is missing or not a list."""
    if not isinstance(x, dict):
        return []
    val = x.get(key)
    if not isinstance(val, list):
        return []
    return val


def _extract_json_block(text: str) -> Optional[Dict]:
    """Extracts a JSON object from a text block."""
    if not text:
        return None
    # Look for JSON within ```json ``` or {} brackets
    import re

    # Try to find JSON within code blocks first
    json_match = re.search(r"```(?:json)?\s*({.*?})\s*```", text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except json.JSONDecodeError:
            pass

    # Try to find JSON within curly braces
    brace_start = text.find("{")
    if brace_start != -1:
        # Find the matching closing brace
        brace_count = 0
        for i, char in enumerate(text[brace_start:], brace_start):
            if char == "{":
                brace_count += 1
            elif char == "}":
                brace_count -= 1
                if brace_count == 0:
                    json_str = text[brace_start : i + 1]
                    try:
                        return json.loads(json_str)
                    except json.JSONDecodeError:
                        break

    return None


def _compose_messages(system_prompt: str, user_prompt: str) -> List[Dict[str, str]]:
    """Composes system and user messages for OpenAI API."""
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def _hash_text(text: str) -> str:
    """Create a hash of the text."""
    return hashlib.md5(text.encode()).hexdigest()


# Validation rules for different content types
VALIDATION_RULES = {
    "generic": {
        "max_length": 10000,
        "required_sections": ["Overview", "Scope"],
        "required_keywords": ["purpose", "requirements"],
        "forbidden_patterns": [r"\bTODO\b", r"\bFIXME\b"],
        "min_complexity": 20,
    },
    "protocol": {
        "max_length": 5000,
        "required_sections": ["Overview", "Scope", "Procedure"],
        "required_keywords": ["procedure", "guidelines", "requirements"],
        "forbidden_patterns": [r"\bTODO\b", r"\bFIXME\b"],
        "min_complexity": 30,
    },
    "security": {
        "max_length": 8000,
        "required_sections": ["Overview", "Threats", "Mitigation", "Response"],
        "required_keywords": [
            "vulnerability",
            "threat",
            "mitigation",
            "access control",
        ],
        "forbidden_patterns": [r"\bTODO\b", r"\bFIXME\b"],
        "min_complexity": 50,
    },
    "code": {
        "max_length": 10000,
        "required_sections": ["Overview", "Implementation", "Examples"],
        "required_keywords": ["function", "class", "method", "implementation"],
        "forbidden_patterns": [r"\bTODO\b", r"\bFIXME\b"],
        "min_complexity": 40,
    },
    "legal": {
        "max_length": 15000,
        "required_sections": ["Parties", "Terms", "Conditions", "Liabilities"],
        "required_keywords": ["agreement", "party", "liability", "compliance"],
        "forbidden_patterns": [r"\bTODO\b", r"\bFIXME\b"],
        "min_complexity": 60,
    },
}

# Report templates
REPORT_TEMPLATES = {
    "executive_summary": {
        "name": "Executive Summary Report",
        "description": "High-level summary of findings and recommendations",
        "format": "markdown",
        "sections": [
            "Executive Summary",
            "Key Findings",
            "Recommendations",
            "Conclusion",
        ],
    },
    "technical_analysis": {
        "name": "Technical Analysis Report",
        "description": "Detailed technical analysis with implementation details",
        "format": "markdown",
        "sections": [
            "Introduction",
            "Methodology",
            "Analysis",
            "Results",
            "Implementation",
            "Conclusion",
        ],
    },
    "security_audit": {
        "name": "Security Audit Report",
        "description": "Comprehensive security audit with vulnerabilities and remediation",
        "format": "markdown",
        "sections": [
            "Audit Scope",
            "Methodology",
            "Vulnerabilities",
            "Risk Assessment",
            "Remediation",
            "Conclusion",
        ],
    },
    "compliance_review": {
        "name": "Compliance Review Report",
        "description": "Review of compliance with regulations and standards",
        "format": "markdown",
        "sections": [
            "Compliance Framework",
            "Assessment",
            "Findings",
            "Remediation Plan",
            "Conclusion",
        ],
    },
    "performance_evaluation": {
        "name": "Performance Evaluation Report",
        "description": "Evaluation of system or process performance",
        "format": "markdown",
        "sections": [
            "Objectives",
            "Metrics",
            "Analysis",
            "Findings",
            "Improvements",
            "Conclusion",
        ],
    },
}

# AI prompt templates
APPROVAL_PROMPT = """You are an evaluator assessing the quality of a Standard Operating Procedure (SOP).
Please evaluate the provided SOP according to these criteria:
1. Clarity: Is the SOP clear and unambiguous?
2. Completeness: Does the SOP cover all necessary steps?
3. Feasibility: Is the SOP practically implementable?
4. Safety: Does the SOP avoid hazardous or dangerous instructions?
5. Compliance: Does the SOP adhere to best practices?

Respond in JSON format with:
- \"verdict\": \"APPROVED\" or \"REJECTED\"
- \"score\": 0-100 (numerical score)
- \"reasons\": [array of brief reason strings for the verdict]
- \"suggestions\": [array of improvement suggestions if any]

SOP: {sop}"""

RED_TEAM_CRITIQUE_PROMPT = """You are a critical reviewer examining a Standard Operating Procedure (SOP) for flaws and vulnerabilities.
Your role is to identify potential problems, risks, and weaknesses in the SOP from a red-team perspective.
Focus on finding:
1. Logical errors or gaps in the procedure
2. Security vulnerabilities or risks
3. Ambiguous or unclear instructions
4. Potential for misinterpretation
5. Missing edge cases or exception handling
6. Compliance with best practices

Be specific and constructive in your critique.
{compliance_requirements}

Respond in JSON format with:
- \"issues\": [array of issue objects, each with \"title\", \"description\", \"severity\" (low/medium/high/critical), and \"category\"]
- \"overall_assessment\": \"string with your overall assessment\"

SOP: {sop}"""

BLUE_TEAM_PATCH_PROMPT = """You are an improvement specialist tasked with fixing issues in a Standard Operating Procedure (SOP).
A red team has identified the following issues that need to be addressed:
{critiques}

Your task is to improve the SOP by addressing these issues while preserving its core purpose.
Update the SOP by incorporating fixes for the identified issues.

Respond in JSON format with:
- \"sop\": \"the updated SOP with fixes applied\" 
- \"mitigation_matrix\": [{\"issue\": \"issue title\", \"status\": \"resolved/mitigated/acknowledged\", \"approach\": \"brief description of how it was addressed\"}]
- \"residual_risks\": [\"list of any remaining risks or concerns\"]

Original SOP: {sop}"""

CODE_REVIEW_RED_TEAM_PROMPT = """You are a code reviewer examining a code implementation for flaws and vulnerabilities.
Your role is to identify potential problems, security vulnerabilities, and code quality issues.
Focus on finding:
1. Security vulnerabilities (injection, XSS, auth issues, etc.)
2. Code quality issues (complexity, maintainability, readability)
3. Performance problems (inefficient algorithms, resource leaks)
4. Logic errors or bugs
5. Best practice violations
6. Documentation gaps

{compliance_requirements}

Respond in JSON format with:
- \"issues\": [array of issue objects, each with \"title\", \"description\", \"severity\" (low/medium/high/critical), and \"category\"]
- \"overall_assessment\": \"string with your overall assessment\"

Code: {sop}"""

CODE_REVIEW_BLUE_TEAM_PROMPT = """You are a code improvement specialist tasked with fixing issues in a code implementation.
A red team has identified the following issues that need to be addressed:
{critiques}

Your task is to improve the code by addressing these issues while preserving its core functionality.
Update the code by incorporating fixes for the identified issues, improving security, performance, and quality.

Respond in JSON format with:
- \"sop\": \"the updated code with fixes applied\" 
- \"mitigation_matrix\": [{\"issue\": \"issue title\", \"status\": \"resolved/mitigated/acknowledged\", \"approach\": \"brief description of how it was addressed\"}]
- \"residual_risks\": [\"list of any remaining risks or concerns\"]

Original Code: {sop}"""

PLAN_REVIEW_RED_TEAM_PROMPT = """You are a planning reviewer examining a project plan for flaws and risks.
Your role is to identify potential problems, risks, and weaknesses in the plan.
Focus on finding:
1. Feasibility and resource allocation issues
2. Timeline and dependency problems
3. Risk management gaps
4. Stakeholder alignment issues
5. Budget and scope concerns
6. Success metrics and measurement gaps

{compliance_requirements}

Respond in JSON format with:
- \"issues\": [array of issue objects, each with \"title\", \"description\", \"severity\" (low/medium/high/critical), and \"category\"]
- \"overall_assessment\": \"string with your overall assessment\"

Plan: {sop}"""

PLAN_REVIEW_BLUE_TEAM_PROMPT = """You are a planning improvement specialist tasked with enhancing a project plan.
A red team has identified the following issues that need to be addressed:
{critiques}

Your task is to improve the plan by addressing these issues while preserving its core objectives.
Update the plan by incorporating fixes for the identified issues, improving feasibility and risk management.

Respond in JSON format with:
- \"sop\": \"the updated plan with improvements applied\" 
- \"mitigation_matrix\": [{\"issue\": \"issue title\", \"status\": \"resolved/mitigated/acknowledged\", \"approach\": \"brief description of how it was addressed\"}]
- \"residual_risks\": [\"list of any remaining risks or concerns\"]

Original Plan: {sop}"""


def display_success_message(message: str) -> None:
    """Display a success message in the UI."""
    st.success(message)


def display_error_message(message: str) -> None:
    """Display an error message in the UI."""
    st.error(message)


def display_warning_message(message: str) -> None:
    """Display a warning message in the UI."""
    st.warning(message)


def display_info_message(message: str) -> None:
    """Display an info message in the UI."""
    st.info(message)


# Protocol templates
PROTOCOL_TEMPLATES = {
    "Security Policy": """# Security Policy Template

## Overview
[Brief description of the policy's purpose and scope]

## Scope
[Define what systems, processes, and personnel are covered by this policy]

## Policy Statements
[Specific security requirements and guidelines]

## Roles and Responsibilities
[Define who is responsible for what aspects of the policy]

## Compliance
[How compliance will be measured and enforced]

## Exceptions
[Process for requesting policy exceptions]

## Review and Updates
[How often the policy will be reviewed and updated]""",
    "Standard Operating Procedure": """# Standard Operating Procedure (SOP) Template

## Title
[Name of the procedure]

## Purpose
[Why this procedure exists]

## Scope
[What this procedure covers and who it applies to]

## Responsibilities
[Who is responsible for each step]

## Procedure
1. [First step]
   - [Detailed instructions]
   - [Expected outcomes]
2. [Second step]
   - [Detailed instructions]
   - [Expected outcomes]

## Safety Considerations
[Any safety risks and how to mitigate them]

## Quality Control
[How to ensure quality and consistency]

## Documentation
[What records need to be maintained]

## Revision History
[Track changes to the procedure]""",
    "Incident Response Plan": """# Incident Response Plan Template

## Overview
[Brief description of the plan's purpose]

## Incident Classification
[Types of incidents and severity levels]

## Response Team
[Key personnel and their roles]

## Detection and Reporting
[How incidents are detected and reported]

## Containment
[Immediate actions to limit impact]

## Eradication
[Steps to remove the threat]

## Recovery
[How to restore normal operations]

## Post-Incident Activities
[Lessons learned and plan updates]

## Communication Plan
[Who to notify and when]

## Contact Information
[Key contacts and their availability]""",
    "Software Development Process": """# Software Development Process Template

## Overview
[Brief description of the development process]

## Scope
[What types of projects this process applies to]

## Roles and Responsibilities
- Project Manager: [Responsibilities]
- Developers: [Responsibilities]
- QA Engineers: [Responsibilities]
- DevOps Engineers: [Responsibilities]

## Development Lifecycle
### 1. Requirements Gathering
- [Process for collecting requirements]
- [Stakeholder involvement]

### 2. Design
- [System architecture design]
- [UI/UX design]
- [Database design]

### 3. Implementation
- [Coding standards]
- [Version control practices]
- [Code review process]

### 4. Testing
- [Unit testing]
- [Integration testing]
- [System testing]
- [User acceptance testing]

### 5. Deployment
- [Deployment process]
- [Rollback procedures]
- [Monitoring]

## Quality Assurance
[QA processes and standards]

## Documentation
[Required documentation at each stage]

## Tools and Technologies
[List of tools used in the process]

## Metrics and KPIs
[Key performance indicators to track]

## Review and Improvement
[Process for continuous improvement]""",
    "Data Privacy Policy": """# Data Privacy Policy Template

## Overview
[Statement of commitment to data privacy]

## Scope
[What data and processes this policy covers]

## Legal Compliance
[List of applicable regulations (GDPR, CCPA, etc.)]

## Data Collection
[What data is collected and why]

## Data Usage
[How collected data is used]

## Data Storage
[Where and how data is stored]

## Data Sharing
[When and with whom data may be shared]

## Data Retention
[How long data is retained]

## Individual Rights
- Right to Access
- Right to Rectification
- Right to Eradication
- Right to Restrict Processing
- Right to Data Portability
- Right to Object

## Security Measures
[Technical and organizational measures to protect data]

## Breach Notification
[Process for reporting data breaches]

## Training and Awareness
[Employee training requirements]

## Policy Enforcement
[Consequences for policy violations]

## Review and Updates
[How often the policy is reviewed]""",
    "Business Continuity Plan": """# Business Continuity Plan Template

## Overview
[Purpose and scope of the business continuity plan]

## Risk Assessment
[Identified risks and their potential impact]

## Business Impact Analysis
[Critical business functions and maximum tolerable downtime]

## Recovery Strategies
[Strategies for recovering critical functions]

## Emergency Response
### 1. Incident Declaration
[Criteria for declaring an emergency]

### 2. Emergency Response Team
- Team Members: [List]
- Contact Information: [Details]
- Roles and Responsibilities: [Details]

### 3. Communication Plan
[Internal and external communication procedures]

## Recovery Procedures
### Critical Function 1
- Recovery Steps: [Detailed steps]
- Resources Required: [List]
- Recovery Time Objective: [Timeframe]

### Critical Function 2
- Recovery Steps: [Detailed steps]
- Resources Required: [List]
- Recovery Time Objective: [Timeframe]

## Plan Testing and Maintenance
[Testing schedule and procedures]

## Training and Awareness
[Training requirements for personnel]

## Plan Distribution
[List of plan recipients]

## Plan Activation and Deactivation
[Criteria and procedures for plan activation and deactivation]""",
    "API Security Review Checklist": """# API Security Review Checklist Template

## Overview
[Description of the API and its purpose]

## Authentication
- [ ] Authentication mechanism implemented
- [ ] Strong password policies enforced
- [ ] Multi-factor authentication supported
- [ ] Session management secure

## Authorization
- [ ] Role-based access control implemented
- [ ] Permissions properly configured
- [ ] Least privilege principle applied
- [ ] Access controls tested

## Input Validation
- [ ] All inputs validated
- [ ] SQL injection protection implemented
- [ ] Cross-site scripting (XSS) prevention
- [ ] File upload restrictions in place

## Data Protection
- [ ] Data encryption in transit (TLS)
- [ ] Data encryption at rest
- [ ] Sensitive data masked in logs
- [ ] Personal data handling compliant

## Error Handling
- [ ] Descriptive error messages suppressed
- [ ] Error logging implemented
- [ ] Exception handling in place
- [ ] Stack traces not exposed

## Rate Limiting
- [ ] Rate limiting implemented
- [ ] Throttling configured
- [ ] Brute force protection
- [ ] DDoS protection measures

## Security Headers
- [ ] Content Security Policy (CSP) implemented
- [ ] X-Frame-Options set
- [ ] X-Content-Type-Options set
- [ ] Strict-Transport-Security configured

## API Gateway Security
- [ ] API gateway configured
- [ ] Traffic monitoring enabled
- [ ] Threat detection implemented
- [ ] Request/response filtering

## Third-Party Dependencies
- [ ] Dependencies regularly updated
- [ ] Vulnerability scanning performed
- [ ] Security patches applied
- [ ] Dependency security monitoring

## Logging and Monitoring
- [ ] Security events logged
- [ ] Audit trail maintained
- [ ] Anomaly detection configured
- [ ] Alerting mechanisms in place

## Compliance
- [ ] GDPR compliance (if applicable)
- [ ] HIPAA compliance (if applicable)
- [ ] PCI DSS compliance (if applicable)
- [ ] Industry-specific requirements""",
    "DevOps Workflow": """# DevOps Workflow Template

## Overview
[Brief description of the DevOps workflow and its objectives]

## Scope
[What systems, applications, and environments this workflow covers]

## Roles and Responsibilities
- DevOps Engineer: [Responsibilities]
- Developers: [Responsibilities]
- QA Engineers: [Responsibilities]
- Security Team: [Responsibilities]

## CI/CD Pipeline
### 1. Code Commit
- Branching strategy: [e.g., GitFlow, GitHub Flow]
- Code review process: [Description]
- Static code analysis: [Tools and criteria]

### 2. Continuous Integration
- Automated build process: [Description]
- Unit test execution: [Process]
- Integration test execution: [Process]
- Security scanning: [Tools and criteria]

### 3. Continuous Deployment
- Deployment environments: [List]
- Deployment approval process: [Description]
- Rollback procedures: [Process]
- Monitoring setup: [Tools and metrics]

## Infrastructure as Code
- Tools used: [e.g., Terraform, CloudFormation]
- Version control: [Repository structure]
- Review process: [Approval workflow]
- Testing strategy: [How infrastructure changes are tested]

## Monitoring and Observability
- Metrics collection: [Tools and what is measured]
- Log aggregation: [Tools and retention policy]
- Alerting thresholds: [What triggers alerts]
- Incident response: [Process for handling alerts]

## Security Practices
- Vulnerability scanning: [Schedule and tools]
- Compliance checks: [Process and tools]
- Secret management: [How secrets are handled]
- Access control: [How access is managed]

## Backup and Recovery
- Backup strategy: [What is backed up and how often]
- Recovery time objectives: [RTO targets]
- Recovery point objectives: [RPO targets]
- Testing schedule: [How often recovery is tested]

## Documentation
- Runbooks: [Location and update process]
- Architecture diagrams: [Location and update process]
- Onboarding guides: [For new team members]

## ...""",
    "Risk Assessment Framework": """# Risk Assessment Framework Template

## Overview
[Purpose and scope of the risk assessment framework]

## Risk Categories
- Operational Risks: [Description]
- Security Risks: [Description]
- Compliance Risks: [Description]
- Financial Risks: [Description]
- Reputational Risks: [Description]

## Risk Assessment Process
### 1. Risk Identification
- Methods: [Brainstorming, historical data, expert interviews, etc.]
- Participants: [Who is involved]
- Frequency: [How often assessments are conducted]

### 2. Risk Analysis
- Qualitative analysis: [Method and criteria]
- Quantitative analysis: [Method and criteria]
- Risk owners: [Who is responsible for each risk]

### 3. Risk Evaluation
- Risk appetite: [Organization's tolerance for risk]
- Risk criteria: [How risks are prioritized]
- Risk matrix: [Likelihood vs Impact matrix]

### 4. Risk Treatment
- Avoidance: [When and how risks are avoided]
- Mitigation: [How risks are reduced]
- Transfer: [How risks are transferred]
- Acceptance: [How risks are accepted]

## Risk Monitoring
- Key risk indicators: [Metrics tracked]
- Reporting frequency: [How often reports are generated]
- Escalation procedures: [When and how risks are escalated]

## Roles and Responsibilities
- Risk Manager: [Responsibilities]
- Risk Owners: [Responsibilities]
- Senior Management: [Responsibilities]

## Documentation
- Risk register: [Format and maintenance]
- Assessment reports: [Template and distribution]
- Action plans: [Format and tracking]

## Review and Updates
- Framework review: [Frequency and process]
- Lessons learned: [How insights are captured]
- Continuous improvement: [Process for implementing changes]""",
    "Disaster Recovery Plan": """# Disaster Recovery Plan Template

## Overview
[Brief description of the disaster recovery plan's purpose]

## Scope
[What systems, applications, and data are covered by this plan]

## Recovery Objectives
- Recovery Time Objective (RTO): [Maximum acceptable downtime]
- Recovery Point Objective (RPO): [Maximum acceptable data loss]

## Critical Systems and Applications
[Identify and prioritize critical systems and applications]

## Disaster Recovery Team
- Team Members: [List]
- Contact Information: [Details]
- Roles and Responsibilities: [Details]

## Disaster Recovery Sites
- Primary Site: [Location and details]
- Backup Site: [Location and details]
- Hot/Warm/Cold Site: [Specify type and capabilities]

## Data Backup and Recovery
- Backup Schedule: [Frequency and methods]
- Backup Storage: [Locations and security]
- Recovery Procedures: [Step-by-step instructions]

## Communication Plan
- Internal Communication: [How to notify employees]
- External Communication: [How to notify customers, vendors, etc.]
- Media Relations: [How to handle media inquiries]
- Law Enforcement: [How to communicate with law enforcement]

## Recovery Procedures
### System Restoration
- [Step-by-step instructions for restoring systems]
- [Required resources and personnel]

### Data Recovery
- [Step-by-step instructions for recovering data]
- [Validation procedures to ensure data integrity]

### Application Recovery
- [Step-by-step instructions for recovering applications]
- [Testing procedures to ensure functionality]

## Plan Testing and Maintenance
- Testing Schedule: [How often the plan will be tested]
- Testing Procedures: [Methods for testing the plan]
- Update Procedures: [How and when the plan will be updated]

## Training and Awareness
- Training Schedule: [How often personnel will be trained]
- Training Materials: [Resources for training personnel]
- Awareness Program: [How to keep personnel informed]""",
    "Change Management Process": """# Change Management Process Template

## Overview
[Brief description of the change management process]

## Scope
[What types of changes are covered by this process]

## Change Categories
- Emergency Changes: [Description and criteria]
- Standard Changes: [Description and criteria]
- Normal Changes: [Description and criteria]
- Major Changes: [Description and criteria]

## Change Management Team
- Change Manager: [Name and contact information]
- Change Advisory Board (CAB): [Members and roles]
- Change Requester: [Role and responsibilities]
- Change Implementer: [Role and responsibilities]

## Change Request Process
### 1. Change Request Submission
- Request Form: [Template and submission process]
- Required Information: [What information must be provided]

### 2. Change Request Review
- Initial Review: [Who conducts the review and criteria]
- Detailed Assessment: [How the change is assessed]
- Risk Assessment: [How risks are identified and evaluated]

### 3. Change Approval
- Approval Authority: [Who has approval authority]
- Approval Criteria: [What criteria are used for approval]
- Approval Process: [How approvals are obtained]

### 4. Change Implementation
- Implementation Plan: [How the change is implemented]
- Implementation Schedule: [When the change is implemented]
- Implementation Team: [Who implements the change]

### 5. Change Closure
- Validation: [How the change is validated]
- Documentation: [How the change is documented]
- Communication: [How stakeholders are informed]

## Change Management Database
- Database Structure: [How change information is organized]
- Data Retention: [How long change information is retained]
- Reporting: [What reports are generated]

## Training and Awareness
- Training Program: [How personnel are trained]
- Awareness Program: [How personnel are kept informed]

## Plan Review and Improvement
- Review Schedule: [How often the process is reviewed]""",
    "Business Impact Analysis": """# Business Impact Analysis Template

## Overview
[Brief description of the business impact analysis]

## Scope
[What business functions and processes are covered]

## Business Functions
[List of critical business functions]

## Dependencies
[Dependencies between business functions]

## Impact Criteria
- Financial Impact: [How financial impact is measured]
- Operational Impact: [How operational impact is measured]
- Legal/Regulatory Impact: [How legal/regulatory impact is measured]
- Reputational Impact: [How reputational impact is measured]

## Recovery Time Objectives
[Recovery time objectives for each business function]

## Recovery Point Objectives
[Recovery point objectives for each business function]

## Minimum Business Continuity Objective
[Minimum business continuity objective for each business function]

## Resource Requirements
[Resources required to recover each business function]

## Interim Recovery Strategies
[Interim recovery strategies for each business function]

## Long-Term Recovery Strategies
[Long-term recovery strategies for each business function]

## Plan Testing and Maintenance
[How the business impact analysis is tested and maintained]

## Review and Approval
[How the business impact analysis is reviewed and approved]""",
    "Data Classification Policy": """# Data Classification Policy Template

## Overview
[Statement of commitment to data classification]

## Scope
[What data and systems this policy covers]

## Data Classification Levels
### 1. Public Data
- Description: [Data that can be freely shared]
- Handling Requirements: [How to handle this data]
- Examples: [Examples of public data]

### 2. Internal Data
- Description: [Data for internal use only]
- Handling Requirements: [How to handle this data]
- Examples: [Examples of internal data]

### 3. Confidential Data
- Description: [Sensitive data requiring protection]
- Handling Requirements: [How to handle this data]
- Examples: [Examples of confidential data]

### 4. Restricted Data
- Description: [Highly sensitive data with strict controls]
- Handling Requirements: [How to handle this data]
- Examples: [Examples of restricted data]

## Data Ownership
- Data Owners: [Who owns different types of data]
- Responsibilities: [What data owners are responsible for]

## Data Handling Procedures
- Storage: [How to store data at each classification level]
- Transmission: [How to transmit data at each classification level]
- Disposal: [How to dispose of data at each classification level]
- Access Control: [Who can access data at each classification level]

## Training and Awareness
[Requirements for data classification training]

## Compliance and Enforcement
[How compliance is measured and enforced]

## Review and Updates
[How often the policy is reviewed]""",
    "Incident Response Communication Plan": """# Incident Response Communication Plan Template

## Overview
[Brief description of the communication plan's purpose]

## Objectives
[What the communication plan aims to achieve]

## Communication Team
- Team Lead: [Name and contact information]
- Spokesperson: [Name and contact information]
- Technical Lead: [Name and contact information]
- Legal Advisor: [Name and contact information]
- HR Representative: [Name and contact information]

## Stakeholder Groups
### 1. Internal Stakeholders
- Employees: [How to communicate with employees]
- Management: [How to communicate with management]
- IT Staff: [How to communicate with IT staff]

### 2. External Stakeholders
- Customers: [How to communicate with customers]
- Vendors: [How to communicate with vendors]
- Regulators: [How to communicate with regulators]
- Media: [How to communicate with media]
- Law Enforcement: [How to communicate with law enforcement]

## Communication Channels
- Email: [When and how to use email]
- Phone: [When and how to use phone]
- Website: [When and how to update website]
- Social Media: [When and how to use social media]
- Press Releases: [When and how to issue press releases]

## Communication Templates
### Initial Notification
[Template for initial incident notification]

### Progress Updates
[Template for progress updates]

### Resolution Notification
[Template for resolution notification]

### Post-Incident Report
[Template for post-incident report]

## Escalation Procedures
[When and how to escalate communications]

## Approval Process
[Who must approve external communications]

## Training and Awareness
[Requirements for communication team training]""",
    "Vulnerability Management Process": """# Vulnerability Management Process Template

## Overview
[Brief description of the vulnerability management process]

## Scope
[What systems, applications, and networks this process covers]

## Roles and Responsibilities
- Vulnerability Manager: [Responsibilities]
- System Owners: [Responsibilities]
- Security Team: [Responsibilities]
- IT Operations: [Responsibilities]

## Vulnerability Management Lifecycle
### 1. Discovery
- Scanning Tools: [Tools used for vulnerability scanning]
- Scanning Schedule: [How often scans are performed]
- Asset Inventory: [How assets are tracked]

### 2. Assessment
- Risk Rating: [How vulnerabilities are rated]
- Impact Analysis: [How impact is assessed]
- Prioritization: [How vulnerabilities are prioritized]

### 3. Remediation
- Patch Management: [How patches are applied]
- Workarounds: [Temporary solutions for unpatched vulnerabilities]
- Exception Process: [When and how to request exceptions]

### 4. Verification
- Re-scanning: [How to verify remediation]
- Penetration Testing: [When and how penetration testing is performed]
- Compliance Checking: [How compliance is verified]

## Tools and Technologies
[List of tools used in the vulnerability management process]

## Reporting
- Dashboard: [What metrics are tracked]
- Executive Reports: [What information is provided to executives]
- Compliance Reports: [What information is provided for compliance]

## Training and Awareness
[Requirements for vulnerability management training]

## Plan Review and Improvement
[How the process is reviewed and improved]""",
}


# Protocol Template Marketplace
PROTOCOL_TEMPLATE_MARKETPLACE = {
    "Security Templates": {
        "NIST Cybersecurity Framework Implementation Guide": {
            "description": "Comprehensive guide for implementing the NIST Cybersecurity Framework",
            "category": "Security",
            "complexity": "Advanced",
            "compliance": ["NIST"],
            "tags": ["cybersecurity", "framework", "risk management"],
            "author": "NIST",
            "rating": 4.8,
            "downloads": 12500,
        },
        "ISO 27001 Information Security Management System": {
            "description": "Template for implementing an ISO 27001 compliant ISMS",
            "category": "Security",
            "complexity": "Advanced",
            "compliance": ["ISO 27001"],
            "tags": ["information security", "ISMS", "compliance"],
            "author": "ISO",
            "rating": 4.7,
            "downloads": 9800,
        },
        "OWASP Top 10 Mitigation Strategies": {
            "description": "Practical strategies for mitigating the OWASP Top 10 web application risks",
            "category": "Security",
            "complexity": "Intermediate",
            "compliance": ["OWASP"],
            "tags": ["web security", "OWASP", "application security"],
            "author": "OWASP Community",
            "rating": 4.9,
            "downloads": 15200,
        },
    },
    "Compliance Templates": {
        "GDPR Data Protection Impact Assessment": {
            "description": "Template for conducting GDPR DPIAs for data processing activities",
            "category": "Compliance",
            "complexity": "Intermediate",
            "compliance": ["GDPR"],
            "tags": ["privacy", "GDPR", "DPIA", "data protection"],
            "author": "EU GDPR Expert Group",
            "rating": 4.6,
            "downloads": 8700,
        },
        "HIPAA Security Rule Compliance Checklist": {
            "description": "Comprehensive checklist for HIPAA Security Rule compliance",
            "category": "Compliance",
            "complexity": "Intermediate",
            "compliance": ["HIPAA"],
            "tags": ["healthcare", "HIPAA", "security", "compliance"],
            "author": "HHS OCR",
            "rating": 4.5,
            "downloads": 7600,
        },
        "SOX IT General Controls Framework": {
            "description": "Framework for implementing SOX-compliant IT general controls",
            "category": "Compliance",
            "complexity": "Advanced",
            "compliance": ["SOX"],
            "tags": ["financial", "SOX", "ITGC", "controls"],
            "author": "SEC Compliance Team",
            "rating": 4.4,
            "downloads": 6500,
        },
    },
    "DevOps Templates": {
        "Kubernetes Security Best Practices": {
            "description": "Security best practices for Kubernetes cluster deployment and management",
            "category": "DevOps",
            "complexity": "Advanced",
            "compliance": [],
            "tags": ["Kubernetes", "container security", "DevSecOps"],
            "author": "CNCF Security SIG",
            "rating": 4.8,
            "downloads": 11200,
        },
        "CI/CD Pipeline Security Checklist": {
            "description": "Security checklist for securing CI/CD pipelines",
            "category": "DevOps",
            "complexity": "Intermediate",
            "compliance": [],
            "tags": ["CI/CD", "pipeline security", "DevSecOps"],
            "author": "DevSecOps Community",
            "rating": 4.7,
            "downloads": 9800,
        },
        "Infrastructure as Code Security Guide": {
            "description": "Guide for securing infrastructure deployed through IaC tools",
            "category": "DevOps",
            "complexity": "Intermediate",
            "compliance": [],
            "tags": ["IaC", "Terraform", "security", "cloud"],
            "author": "Cloud Security Alliance",
            "rating": 4.6,
            "downloads": 8900,
        },
    },
    "Business Templates": {
        "Business Continuity Plan for Remote Work": {
            "description": "Comprehensive BCP tailored for remote and hybrid work environments",
            "category": "Business",
            "complexity": "Intermediate",
            "compliance": [],
            "tags": ["business continuity", "remote work", "pandemic planning"],
            "author": "Business Resilience Institute",
            "rating": 4.5,
            "downloads": 7200,
        },
        "Digital Transformation Roadmap": {
            "description": "Step-by-step roadmap for enterprise digital transformation",
            "category": "Business",
            "complexity": "Advanced",
            "compliance": [],
            "tags": ["digital transformation", "change management", "strategy"],
            "author": "Digital Transformation Experts",
            "rating": 4.7,
            "downloads": 8400,
        },
        "Vendor Risk Management Framework": {
            "description": "Framework for assessing and managing third-party vendor risks",
            "category": "Business",
            "complexity": "Advanced",
            "compliance": [],
            "tags": ["vendor management", "third-party risk", "supply chain"],
            "author": "Risk Management Association",
            "rating": 4.6,
            "downloads": 6800,
        },
    },
}

# Adversarial Testing Presets
ADVERSARIAL_PRESETS = {
    "Security Hardening": {
        "name": "🔐 Security Hardening",
        "description": "Focus on identifying and closing security gaps, enforcing least privilege, and adding comprehensive error handling.",
        "red_team_models": [
            "openai/gpt-4o-mini",
            "anthropic/claude-3-haiku",
            "google/gemini-1.5-flash",
        ],
        "blue_team_models": [
            "openai/gpt-4o",
            "anthropic/claude-3-sonnet",
            "google/gemini-1.5-pro",
        ],
        "min_iter": 5,
        "max_iter": 15,
        "confidence_threshold": 95,
        "review_type": "General SOP",
        "compliance_requirements": "Security best practices, OWASP guidelines, least privilege principle",
        "advanced_settings": {
            "critique_depth": 8,
            "patch_quality": 9,
            "detailed_tracking": True,
            "early_stopping": True,
        },
    },
    "Compliance Focus": {
        "name": "⚖️ Compliance Focus",
        "description": "Ensure protocols meet regulatory requirements with comprehensive auditability.",
        "red_team_models": ["openai/gpt-4o-mini", "mistral/mistral-small-latest"],
        "blue_team_models": ["openai/gpt-4o", "mistral/mistral-medium-latest"],
        "min_iter": 3,
        "max_iter": 10,
        "confidence_threshold": 90,
        "review_type": "General SOP",
        "compliance_requirements": "GDPR, ISO 27001, SOC 2, industry-specific regulations",
        "advanced_settings": {
            "critique_depth": 7,
            "patch_quality": 8,
            "detailed_tracking": True,
            "performance_analytics": True,
        },
    },
    "Operational Efficiency": {
        "name": "⚡ Operational Efficiency",
        "description": "Streamline processes while maintaining effectiveness and clarity.",
        "red_team_models": ["openai/gpt-4o-mini", "meta-llama/llama-3-8b-instruct"],
        "blue_team_models": ["openai/gpt-4o", "meta-llama/llama-3-70b-instruct"],
        "min_iter": 3,
        "max_iter": 12,
        "confidence_threshold": 85,
        "review_type": "General SOP",
        "compliance_requirements": "Process optimization, resource efficiency, clarity standards",
        "advanced_settings": {
            "critique_depth": 6,
            "patch_quality": 7,
            "early_stopping": True,
            "target_complexity": 50,
        },
    },
    "Beginner-Friendly": {
        "name": "👶 Beginner-Friendly",
        "description": "Focus on clarity, simplicity, and completeness for newcomers.",
        "red_team_models": ["openai/gpt-4o-mini", "google/gemini-1.5-flash"],
        "blue_team_models": ["openai/gpt-4o", "google/gemini-1.5-pro"],
        "min_iter": 2,
        "max_iter": 8,
        "confidence_threshold": 80,
        "review_type": "General SOP",
        "compliance_requirements": "Clear language, simple concepts, comprehensive examples",
        "advanced_settings": {
            "critique_depth": 5,
            "patch_quality": 8,
            "target_complexity": 30,
            "target_length": 500,
        },
    },
    "Code Review": {
        "name": "💻 Code Review",
        "description": "Specialized testing for software development protocols and code reviews.",
        "red_team_models": [
            "openai/gpt-4o",
            "anthropic/claude-3-opus",
            "codellama/codellama-70b-instruct",
        ],
        "blue_team_models": [
            "openai/gpt-4o",
            "anthropic/claude-3-sonnet",
            "codellama/codellama-70b-instruct",
        ],
        "min_iter": 3,
        "max_iter": 10,
        "confidence_threshold": 90,
        "review_type": "Code Review",
        "compliance_requirements": "Clean code principles, security best practices, performance optimization",
        "advanced_settings": {
            "critique_depth": 9,
            "patch_quality": 9,
            "detailed_tracking": True,
            "performance_analytics": True,
        },
    },
    "Mission Critical": {
        "name": "🔥 Mission Critical",
        "description": "Maximum rigor for high-stakes protocols requiring the highest assurance.",
        "red_team_models": [
            "openai/gpt-4o",
            "anthropic/claude-3-opus",
            "google/gemini-1.5-pro",
        ],
        "blue_team_models": [
            "openai/gpt-4o",
            "anthropic/claude-3-sonnet",
            "google/gemini-1.5-pro",
        ],
        "min_iter": 10,
        "max_iter": 25,
        "confidence_threshold": 98,
        "review_type": "General SOP",
        "compliance_requirements": "Highest security standards, fault tolerance, disaster recovery",
        "advanced_settings": {
            "critique_depth": 10,
            "patch_quality": 10,
            "detailed_tracking": True,
            "performance_analytics": True,
            "early_stopping": False,
        },
    },
    "AI Safety Review": {
        "name": "🛡️ AI Safety Review",
        "description": "Specialized testing for AI safety considerations, bias detection, and ethical alignment.",
        "red_team_models": [
            "openai/gpt-4o",
            "anthropic/claude-3-opus",
            "google/gemini-1.5-pro",
        ],
        "blue_team_models": [
            "openai/gpt-4o",
            "anthropic/claude-3-sonnet",
            "meta-llama/llama-3-70b-instruct",
        ],
        "min_iter": 5,
        "max_iter": 15,
        "confidence_threshold": 92,
        "review_type": "General SOP",
        "compliance_requirements": "AI safety guidelines, fairness principles, explainability requirements",
        "advanced_settings": {
            "critique_depth": 9,
            "patch_quality": 9,
            "detailed_tracking": True,
            "performance_analytics": True,
            "bias_detection": True,
            "explainability_focus": True,
        },
    },
    "Privacy Protection": {
        "name": "🔒 Privacy Protection",
        "description": "Focus on data privacy protection, consent mechanisms, and regulatory compliance.",
        "red_team_models": [
            "openai/gpt-4o-mini",
            "anthropic/claude-3-haiku",
            "google/gemini-1.5-flash",
        ],
        "blue_team_models": [
            "openai/gpt-4o",
            "anthropic/claude-3-sonnet",
            "google/gemini-1.5-pro",
        ],
        "min_iter": 4,
        "max_iter": 12,
        "confidence_threshold": 93,
        "review_type": "General SOP",
        "compliance_requirements": "GDPR, CCPA, PIPEDA, data minimization principles",
        "advanced_settings": {
            "critique_depth": 8,
            "patch_quality": 8,
            "detailed_tracking": True,
            "privacy_by_design": True,
            "consent_mechanisms": True,
        },
    },
}

DEFAULTS = {
    "provider": "OpenAI",
    "api_key": "",
    "base_url": PROVIDERS["OpenAI"]["base"],
    "model": PROVIDERS["OpenAI"]["model"],
    "extra_headers": "{}",
    "max_tokens": 4096,
    "temperature": 0.7,
    "top_p": 1.0,
    "frequency_penalty": 0.0,
    "presence_penalty": 0.0,
    "seed": "",
    "protocol_text": "",
    "system_prompt": "You are an assistant that makes the draft airtight, precise, and foolproof.",
    "evaluator_system_prompt": "You are a trivial evaluator that accepts everything.",
    "max_iterations": 20,
    "population_size": 1,
    "num_islands": 1,
    "checkpoint_interval": 5,
    "elite_ratio": 1.0,
    "exploration_ratio": 0.0,
    "exploitation_ratio": 0.0,
    "archive_size": 0,
    "evolution_running": False,
    "evolution_log": [],
    "evolution_current_best": "",
    "evolution_stop_flag": False,
    "openrouter_key": "",
    "red_team_models": [],
    "blue_team_models": [],
    "adversarial_running": False,
    "adversarial_results": {},
    "adversarial_status_message": "Idle.",
    "adversarial_log": [],
    "adversarial_stop_flag": False,
    "adversarial_cost_estimate_usd": 0.0,
    "adversarial_total_tokens_prompt": 0,
    "adversarial_total_tokens_completion": 0,
    "adversarial_min_iter": 3,
    "adversarial_max_iter": 10,
    "adversarial_confidence": 95,
    "adversarial_max_tokens": 8000,
    "adversarial_max_workers": 6,
    "adversarial_force_json": True,
    "adversarial_seed": "",
    "adversarial_rotation_strategy": "None",
    "adversarial_red_team_sample_size": 3,
    "adversarial_blue_team_sample_size": 3,
    "adversarial_model_performance": {},
    "adversarial_confidence_history": [],
    "adversarial_staged_rotation_config": "",
    "compliance_requirements": "",
    # Collaborative features
    "project_name": "Untitled Project",
    "collaborators": [],
    "comments": [],
    "protocol_versions": [],
    "current_version_id": "",
    "tags": [],
    "project_description": "",
    # Tutorial and onboarding
    "tutorial_completed": False,
    "current_tutorial_step": 0,
    # User preferences and theme
    "theme": "light",
    "user_preferences": {
        "theme": "light",
        "font_size": "medium",
        "auto_save": True,
        "show_notifications": True,
        "notification_preferences": {"email": True, "push": True, "slack": False},
    },
}

for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v


def reset_defaults():
    p = st.session_state.provider
    if p in PROVIDERS:
        st.session_state.base_url = PROVIDERS[p].get("base", "")
        st.session_state.model = PROVIDERS[p].get("model") or ""
    st.session_state.api_key = ""
    st.session_state.extra_headers = "{}"


def save_user_preferences():
    """
    Save user preferences to session state.
    """
    try:
        if "user_preferences" not in st.session_state:
            st.session_state.user_preferences = {}

        # Update preferences from UI elements
        st.session_state.user_preferences["theme"] = st.session_state.get(
            "theme", "light"
        )
        st.session_state.user_preferences["font_size"] = st.session_state.get(
            "font_size", "medium"
        )
        st.session_state.user_preferences["auto_save"] = st.session_state.get(
            "auto_save", True
        )

        return True
    except Exception as e:
        st.error(f"Error saving user preferences: {e}")
        return False


def toggle_theme():
    """
    Toggle between light and dark theme.
    """
    current_theme = st.session_state.get("theme", "light")
    new_theme = "dark" if current_theme == "light" else "light"
    st.session_state.theme = new_theme

    # Update user preferences
    if "user_preferences" not in st.session_state:
        st.session_state.user_preferences = {}
    st.session_state.user_preferences["theme"] = new_theme

    return new_theme
