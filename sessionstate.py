import streamlit as st
import threading
from typing import Any, Callable, Dict, List, Optional, Tuple
import uuid
from datetime import datetime
import json
import hashlib
import re
from analytics import generate_advanced_analytics, analyze_plan_quality
from providers import PROVIDERS
from config_data import CONFIG_PROFILES
from config_data import CONFIG_PROFILES

# Placeholder for missing functions/variables. These will be searched for or created.
# If you have the original definitions for these, please provide them.
def calculate_protocol_complexity(protocol_text: str) -> Dict:
    words = protocol_text.split()
    word_count = len(words)
    sentences = re.split(r'[.!?]', protocol_text)
    sentence_count = len([s for s in sentences if s.strip()])
    paragraphs = protocol_text.split('\n\n')
    paragraph_count = len([p for p in paragraphs if p.strip()])
    unique_words = len(set(w.lower() for w in words))
    avg_sentence_length = word_count / max(1, sentence_count)
    
    # Simple complexity score: more words, longer sentences, fewer unique words = higher complexity
    complexity_score = (word_count / 100) + (avg_sentence_length * 2) - (unique_words / 50)
    complexity_score = max(0, min(100, complexity_score)) # Clamp between 0 and 100

    return {
        "word_count": word_count,
        "sentence_count": sentence_count,
        "paragraph_count": paragraph_count,
        "complexity_score": round(complexity_score, 2),
        "unique_words": unique_words,
        "avg_sentence_length": round(avg_sentence_length, 2)
    }

def extract_protocol_structure(protocol_text: str) -> Dict:
    has_headers = bool(re.search(r'^#+\s', protocol_text, re.MULTILINE))
    has_numbered_steps = bool(re.search(r'^\d+\.\s', protocol_text, re.MULTILINE))
    has_bullet_points = bool(re.search(r'^-+\s', protocol_text, re.MULTILINE))
    has_preconditions = "preconditions" in protocol_text.lower()
    has_postconditions = "postconditions" in protocol_text.lower()
    has_error_handling = "error handling" in protocol_text.lower() or "exception handling" in protocol_text.lower()
    section_count = len(re.findall(r'^#+\s', protocol_text, re.MULTILINE)) + len(re.findall(r'^.*\\n[=]{3,}', protocol_text, re.MULTILINE))

    return {
        "has_headers": has_headers,
        "has_numbered_steps": has_numbered_steps,
        "has_bullet_points": has_bullet_points,
        "has_preconditions": has_preconditions,
        "has_postconditions": has_postconditions,
        "has_error_handling": has_error_handling,
        "section_count": section_count
    }

def generate_protocol_recommendations(protocol_text: str) -> List[str]:
    recommendations = []
    if len(protocol_text.split()) < 50:
        recommendations.append("Consider expanding the protocol with more details.")
    if "todo" in protocol_text.lower():
        recommendations.append("Address any 'TODO' items in the protocol.")
    if not re.search(r'^\s*#+\s', protocol_text, re.MULTILINE):
        recommendations.append("Add clear section headers to improve readability.")
    if not recommendations:
        """
Modular Session State for OpenEvolve
This file has been refactored into a modular architecture. 
Please use session_manager.py to access all functionality.
"""
# This file is maintained for backward compatibility
# All functionality has been moved to modular files
from .session_manager import *

# Additional compatibility imports if needed
from .session_utils import (
    _clamp,
    _rand_jitter_ms,
    _approx_tokens,
    _cost_estimate,
    safe_int,
    safe_float,
    _safe_list,
    _extract_json_block,
    _compose_messages,
    APPROVAL_PROMPT,
    RED_TEAM_CRITIQUE_PROMPT,
    BLUE_TEAM_PATCH_PROMPT,
    CODE_REVIEW_RED_TEAM_PROMPT,
    CODE_REVIEW_BLUE_TEAM_PROMPT,
    PLAN_REVIEW_RED_TEAM_PROMPT,
    PLAN_REVIEW_BLUE_TEAM_PROMPT,
    VALIDATION_RULES
)

# Import all functions from modular manager files for backward compatibility
from .version_control import (
    get_version_by_id,
    compare_versions,
    get_version_count,
    get_current_version,
    create_new_version,
    load_version,
    get_version_history,
    get_version_by_name,
    get_version_by_timestamp,
    delete_version,
    branch_version,
    get_version_timeline,
    render_version_timeline
)

from .collaboration_manager import (
    add_notification,
    get_unread_notifications,
    mark_notification_as_read,
    add_collaborator,
    remove_collaborator,
    get_collaborators,
    update_collaborator_role,
    initialize_collaborative_session,
    join_collaborative_session,
    leave_collaborative_session,
    apply_edit_operation,
    detect_conflicts,
    resolve_conflict,
    get_session_state,
    synchronize_document,
    render_collaborative_editor_ui
)

from .export_import_manager import (
    export_to_json,
    import_from_json,
    export_to_markdown,
    export_to_text,
    export_protocol_with_history,
    validate_import_data,
    export_project,
    export_project_detailed,
    import_project,
    generate_shareable_link,
    export_protocol_as_template
)

from .template_manager import (
    list_template_categories,
    list_templates_in_category,
    get_template_details,
    search_templates,
    get_popular_templates,
    get_top_rated_templates,
    render_template_marketplace_ui,
    add_custom_template,
    get_all_templates,
    get_template_usage_stats
)

from .analytics_manager import (
    generate_ai_insights,
    render_ai_insights_dashboard,
    generate_advanced_analytics,
    calculate_model_performance_metrics
)

from .validation_manager import (
    add_validation_rule,
    update_validation_rule,
    remove_validation_rule,
    list_validation_rules,
    get_validation_rule,
    validate_content_against_custom_rules,
    run_compliance_check
)

from .content_manager import (
    list_protocol_templates,
    load_protocol_template,
    export_protocol_as_template,
    validate_protocol,
    render_validation_results,
    list_report_templates,
    get_report_template_details,
    generate_custom_report,
    generate_content_summary
)
from .analytics_manager import (
    calculate_model_performance_metrics
)
from .integrations import (
    send_discord_notification,
    send_msteams_notification,
    send_generic_webhook,
    authenticate_github,
    list_github_repositories,
    create_github_branch,
    commit_to_github,
    get_github_commit_history,
    render_github_integration_ui,
    link_github_repository,
    unlink_github_repository,
    list_linked_github_repositories,
    save_protocol_generation_to_github,
    get_protocol_generations_from_github,
    render_github_branching_ui,
    render_remote_storage_ui
)
from .rbac import (
    get_user_role,
    has_permission,
    assign_role
)
# Protocol templates
PROTOCOL_TEMPLATES = {
    "Security Policy": """# Security Policy Template\n\n## Overview\n[Brief description of the policy's purpose and scope]\n\n## Scope\n[Define what systems, processes, and personnel are covered by this policy]\n\n## Policy Statements\n[Specific security requirements and guidelines]\n\n## Roles and Responsibilities\n[Define who is responsible for what aspects of the policy]\n\n## Compliance\n[How compliance will be measured and enforced]\n\n## Exceptions\n[Process for requesting policy exceptions]\n\n## Review and Updates\n[How often the policy will be reviewed and updated]""",
    
    "Standard Operating Procedure": """# Standard Operating Procedure (SOP) Template\n\n## Title\n[Name of the procedure]\n\n## Purpose\n[Why this procedure exists]\n\n## Scope\n[What this procedure covers and who it applies to]\n\n## Responsibilities\n[Who is responsible for each step]\n\n## Procedure\n1. [First step]\n   - [Detailed instructions]\n   - [Expected outcomes]\n2. [Second step]\n   - [Detailed instructions]\n   - [Expected outcomes]\n\n## Safety Considerations\n[Any safety risks and how to mitigate them]\n\n## Quality Control\n[How to ensure quality and consistency]\n\n## Documentation\n[What records need to be maintained]\n\n## Revision History\n[Track changes to the procedure]""",
    
    "Incident Response Plan": """# Incident Response Plan Template\n\n## Overview\n[Brief description of the plan's purpose]\n\n## Incident Classification\n[Types of incidents and severity levels]\n\n## Response Team\n[Key personnel and their roles]\n\n## Detection and Reporting\n[How incidents are detected and reported]\n\n## Containment\n[Immediate actions to limit impact]\n\n## Eradication\n[Steps to remove the threat]\n\n## Recovery\n[How to restore normal operations]\n\n## Post-Incident Activities\n[Lessons learned and plan updates]\n\n## Communication Plan\n[Who to notify and when]\n\n## Contact Information\n[Key contacts and their availability]""",
    
    "Software Development Process": """# Software Development Process Template\n\n## Overview\n[Brief description of the development process]\n\n## Scope\n[What types of projects this process applies to]\n\n## Roles and Responsibilities\n- Project Manager: [Responsibilities]\n- Developers: [Responsibilities]\n- QA Engineers: [Responsibilities]\n- DevOps Engineers: [Responsibilities]\n\n## Development Lifecycle\n### 1. Requirements Gathering\n- [Process for collecting requirements]\n- [Stakeholder involvement]\n\n### 2. Design\n- [System architecture design]\n- [UI/UX design]\n- [Database design]\n\n### 3. Implementation\n- [Coding standards]\n- [Version control practices]\n- [Code review process]\n\n### 4. Testing\n- [Unit testing]\n- [Integration testing]\n- [System testing]\n- [User acceptance testing]\n\n### 5. Deployment\n- [Deployment process]\n- [Rollback procedures]\n- [Monitoring]\n\n## Quality Assurance\n[QA processes and standards]\n\n## Documentation\n[Required documentation at each stage]\n\n## Tools and Technologies\n[List of tools used in the process]\n\n## Metrics and KPIs\n[Key performance indicators to track]\n\n## Review and Improvement\n[Process for continuous improvement]""",
    
    "Data Privacy Policy": """# Data Privacy Policy Template\n\n## Overview\n[Statement of commitment to data privacy]\n\n## Scope\n[What data and processes this policy covers]\n\n## Legal Compliance\n[List of applicable regulations (GDPR, CCPA, etc.)]\n\n## Data Collection\n[What data is collected and why]\n\n## Data Usage\n[How collected data is used]\n\n## Data Storage\n[Where and how data is stored]\n\n## Data Sharing\n[When and with whom data may be shared]\n\n## Data Retention\n[How long data is retained]\n\n## Individual Rights\n- Right to Access\n- Right to Rectification\n- Right to Eradication\n- Right to Restrict Processing\n- Right to Data Portability\n- Right to Object\n\n## Security Measures\n[Technical and organizational measures to protect data]\n\n## Breach Notification\n[Process for reporting data breaches]\n\n## Training and Awareness\n[Employee training requirements]\n\n## Policy Enforcement\n[Consequences for policy violations]\n\n## Review and Updates\n[How often the policy is reviewed]""",
    
    "Business Continuity Plan": """# Business Continuity Plan Template\n\n## Overview\n[Purpose and scope of the business continuity plan]\n\n## Risk Assessment\n[Identified risks and their potential impact]\n\n## Business Impact Analysis\n[Critical business functions and maximum tolerable downtime]\n\n## Recovery Strategies\n[Strategies for recovering critical functions]\n\n## Emergency Response\n### 1. Incident Declaration\n[Criteria for declaring an emergency]\n\n### 2. Emergency Response Team\n- Team Members: [List]\n- Contact Information: [Details]\n- Roles and Responsibilities: [Details]\n\n### 3. Communication Plan\n[Internal and external communication procedures]\n\n## Recovery Procedures\n### Critical Function 1\n- Recovery Steps: [Detailed steps]\n- Resources Required: [List]\n- Recovery Time Objective: [Timeframe]\n\n### Critical Function 2\n- Recovery Steps: [Detailed steps]\n- Resources Required: [List]\n- Recovery Time Objective: [Timeframe]\n\n## Plan Testing and Maintenance\n[Testing schedule and procedures]\n\n## Training and Awareness\n[Training requirements for personnel]\n\n## Plan Distribution\n[List of plan recipients]\n\n## Plan Activation and Deactivation\n[Criteria and procedures for plan activation and deactivation]""",
    
    "API Security Review Checklist": """# API Security Review Checklist Template\n\n## Overview\n[Description of the API and its purpose]\n\n## Authentication\n- [ ] Authentication mechanism implemented\n- [ ] Strong password policies enforced\n- [ ] Multi-factor authentication supported\n- [ ] Session management secure\n\n## Authorization\n- [ ] Role-based access control implemented\n- [ ] Permissions properly configured\n- [ ] Least privilege principle applied\n- [ ] Access controls tested\n\n## Input Validation\n- [ ] All inputs validated\n- [ ] SQL injection protection implemented\n- [ ] Cross-site scripting (XSS) prevention\n- [ ] File upload restrictions in place\n\n## Data Protection\n- [ ] Data encryption in transit (TLS)\n- [ ] Data encryption at rest\n- [ ] Sensitive data masked in logs\n- [ ] Personal data handling compliant\n\n## Error Handling\n- [ ] Descriptive error messages suppressed\n- [ ] Error logging implemented\n- [ ] Exception handling in place\n- [ ] Stack traces not exposed\n\n## Rate Limiting\n- [ ] Rate limiting implemented\n- [ ] Throttling configured\n- [ ] Brute force protection\n- [ ] DDoS protection measures\n\n## Security Headers\n- [ ] Content Security Policy (CSP) implemented\n- [ ] X-Frame-Options set\n- [ ] X-Content-Type-Options set\n- [ ] Strict-Transport-Security configured\n\n## API Gateway Security\n- [ ] API gateway configured\n- [ ] Traffic monitoring enabled\n- [ ] Threat detection implemented\n- [ ] Request/response filtering\n\n## Third-Party Dependencies\n- [ ] Dependencies regularly updated\n- [ ] Vulnerability scanning performed\n- [ ] Security patches applied\n- [ ] Dependency security monitoring\n\n## Logging and Monitoring\n- [ ] Security events logged\n- [ ] Audit trail maintained\n- [ ] Anomaly detection configured\n- [ ] Alerting mechanisms in place\n\n## Compliance\n- [ ] GDPR compliance (if applicable)\n- [ ] HIPAA compliance (if applicable)\n- [ ] PCI DSS compliance (if applicable)\n- [ ] Industry-specific regulations met\n\n## Review and Approval\n- Security Reviewer: [Name]\n- Review Date: [Date]\n- Approval Status: [Approved/Rejected/Pending]\n- Notes: [Additional comments]""",
    
    "DevOps Workflow": """# DevOps Workflow Template\n\n## Overview\n[Brief description of the DevOps workflow and its objectives]\n\n## Scope\n[What systems, applications, and environments this workflow covers]\n\n## Roles and Responsibilities\n- DevOps Engineer: [Responsibilities]\n- Developers: [Responsibilities]\n- QA Engineers: [Responsibilities]\n- Security Team: [Responsibilities]\n\n## CI/CD Pipeline\n### 1. Code Commit\n- Branching strategy: [e.g., GitFlow, GitHub Flow]\n- Code review process: [Description]\n- Static code analysis: [Tools and criteria]\n\n### 2. Continuous Integration\n- Automated build process: [Description]\n- Unit test execution: [Process]\n- Integration test execution: [Process]\n- Security scanning: [Tools and criteria]\n\n### 3. Continuous Deployment\n- Deployment environments: [List]\n- Deployment approval process: [Description]\n- Rollback procedures: [Process]\n- Monitoring setup: [Tools and metrics]\n\n## Infrastructure as Code\n- Tools used: [e.g., Terraform, CloudFormation]\n- Version control: [Repository structure]\n- Review process: [Approval workflow]\n- Testing strategy: [How infrastructure changes are tested]\n\n## Monitoring and Observability\n- Metrics collection: [Tools and what is measured]\n- Log aggregation: [Tools and retention policy]\n- Alerting thresholds: [What triggers alerts]\n- Incident response: [Process for handling alerts]\n\n## Security Practices\n- Vulnerability scanning: [Schedule and tools]\n- Compliance checks: [Process and tools]\n- Secret management: [How secrets are handled]\n- Access control: [How access is managed]\n\n## Backup and Recovery\n- Backup strategy: [What is backed up and how often]\n- Recovery time objectives: [RTO targets]\n- Recovery point objectives: [RPO targets]\n- Testing schedule: [How often recovery is tested]\n\n## Documentation\n- Runbooks: [Location and update process]\n- Architecture diagrams: [Location and update process]\n- Onboarding guides: [For new team members]\n\n## Review and Improvement\n- Retrospectives: [Schedule and process]\n- KPI tracking: [Metrics monitored]\n- Continuous improvement: [Process for implementing changes]""",
    
    "Risk Assessment Framework": """# Risk Assessment Framework Template\n\n## Overview\n[Purpose and scope of the risk assessment framework]\n\n## Risk Categories\n- Operational Risks: [Description]\n- Security Risks: [Description]\n- Compliance Risks: [Description]\n- Financial Risks: [Description]\n- Reputational Risks: [Description]\n\n## Risk Assessment Process\n### 1. Risk Identification\n- Methods: [Brainstorming, historical data, expert interviews, etc.]\n- Participants: [Who is involved]\n- Frequency: [How often assessments are conducted]\n\n### 2. Risk Analysis\n- Qualitative analysis: [Method and criteria]\n- Quantitative analysis: [Method and criteria]\n- Risk owners: [Who is responsible for each risk]\n\n### 3. Risk Evaluation\n- Risk appetite: [Organization's tolerance for risk]\n- Risk criteria: [How risks are prioritized]\n- Risk matrix: [Likelihood vs Impact matrix]\n
### 4. Risk Treatment\n- Avoidance: [When and how risks are avoided]\n- Mitigation: [How risks are reduced]\n- Transfer: [How risks are transferred]\n- Acceptance: [How risks are accepted]\n\n## Risk Monitoring\n- Key risk indicators: [Metrics tracked]\n- Reporting frequency: [How often reports are generated]\n- Escalation procedures: [When and how risks are escalated]\n\n## Roles and Responsibilities\n- Risk Manager: [Responsibilities]\n- Risk Owners: [Responsibilities]\n- Senior Management: [Responsibilities]\n\n## Documentation\n- Risk register: [Format and maintenance]\n- Assessment reports: [Template and distribution]\n- Action plans: [Format and tracking]\n\n## Review and Updates\n- Framework review: [Frequency and process]\n- Lessons learned: [How insights are captured]\n- Continuous improvement: [Process for implementing changes]""",
    
    "Disaster Recovery Plan": """# Disaster Recovery Plan Template\n\n## Overview\n[Brief description of the disaster recovery plan's purpose]\n\n## Scope\n[What systems, applications, and data are covered by this plan]\n\n## Recovery Objectives\n- Recovery Time Objective (RTO): [Maximum acceptable downtime]\n- Recovery Point Objective (RPO): [Maximum acceptable data loss]\n\n## Critical Systems and Applications\n[Identify and prioritize critical systems and applications]\n\n## Disaster Recovery Team\n- Team Members: [List]\n- Contact Information: [Details]\n- Roles and Responsibilities: [Details]\n\n## Disaster Recovery Sites\n- Primary Site: [Location and details]\n- Backup Site: [Location and details]\n- Hot/Warm/Cold Site: [Specify type and capabilities]\n\n## Data Backup and Recovery\n- Backup Schedule: [Frequency and methods]\n- Backup Storage: [Locations and security]\n- Recovery Procedures: [Step-by-step instructions]\n\n## Communication Plan\n- Internal Communication: [How to notify employees]\n- External Communication: [How to notify customers, vendors, etc.]\n- Media Relations: [How to handle media inquiries]\n- Law Enforcement: [How to communicate with law enforcement]\n\n## Recovery Procedures\n### System Restoration\n- [Step-by-step instructions for restoring systems]\n- [Required resources and personnel]\n\n### Data Recovery\n- [Step-by-step instructions for recovering data]\n- [Validation procedures to ensure data integrity]\n\n### Application Recovery\n- [Step-by-step instructions for recovering applications]\n- [Testing procedures to ensure functionality]\n\n## Plan Testing and Maintenance\n- Testing Schedule: [How often the plan will be tested]\n- Testing Procedures: [Methods for testing the plan]\n- Update Procedures: [How and when the plan will be updated]\n\n## Training and Awareness\n- Training Schedule: [How often personnel will be trained]\n- Training Materials: [Resources for training personnel]\n- Awareness Program: [How to keep personnel informed]\n\n## Plan Distribution\n[List of plan recipients and distribution methods]\n\n## Plan Activation and Deactivation\n- Activation Criteria: [When to activate the plan]\n- Deactivation Criteria: [When to deactivate the plan]\n- Activation Procedures: [How to activate the plan]\n- Deactivation Procedures: [How to deactivate the plan]""",
    
    "Change Management Process": """# Change Management Process Template\n\n## Overview\n[Brief description of the change management process]\n\n## Scope\n[What types of changes are covered by this process]\n\n## Change Categories\n- Emergency Changes: [Description and criteria]\n- Standard Changes: [Description and criteria]\n- Normal Changes: [Description and criteria]\n- Major Changes: [Description and criteria]\n\n## Change Management Team\n- Change Manager: [Name and contact information]\n- Change Advisory Board (CAB): [Members and roles]\n- Change Requester: [Role and responsibilities]\n- Change Implementer: [Role and responsibilities]\n\n## Change Request Process\n### 1. Change Request Submission\n- Request Form: [Template and submission process]\n- Required Information: [What information must be provided]\n\n### 2. Change Request Review\n- Initial Review: [Who conducts the review and criteria]\n- Detailed Assessment: [How the change is assessed]\n- Risk Assessment: [How risks are identified and evaluated]\n\n### 3. Change Approval\n- Approval Authority: [Who has approval authority]\n- Approval Criteria: [What criteria are used for approval]\n- Approval Process: [How approvals are obtained]\n\n### 4. Change Implementation\n- Implementation Plan: [How the change is implemented]\n- Implementation Schedule: [When the change is implemented]\n- Implementation Team: [Who implements the change]\n\n### 5. Change Closure\n- Validation: [How the change is validated]\n- Documentation: [How the change is documented]\n- Communication: [How stakeholders are informed]\n\n## Change Management Database\n- Database Structure: [How change information is organized]\n- Data Retention: [How long change information is retained]\n- Reporting: [What reports are generated]\n\n## Training and Awareness\n- Training Program: [How personnel are trained]\n- Awareness Program: [How personnel are kept informed]\n\n## Plan Review and Improvement\n- Review Schedule: [How often the process is reviewed]\n- Improvement Process: [How improvements are made]""",
    
    "Business Impact Analysis": """# Business Impact Analysis Template\n\n## Overview\n[Brief description of the business impact analysis]\n\n## Scope\n[What business functions and processes are covered]\n\n## Business Functions\n[List of critical business functions]\n\n## Dependencies\n[Dependencies between business functions]\n\n## Impact Criteria\n- Financial Impact: [How financial impact is measured]\n- Operational Impact: [How operational impact is measured]\n- Legal/Regulatory Impact: [How legal/regulatory impact is measured]\n- Reputational Impact: [How reputational impact is measured]\n\n## Recovery Time Objectives\n[Recovery time objectives for each business function]\n\n## Recovery Point Objectives\n[Recovery point objectives for each business function]\n\n## Minimum Business Continuity Objective\n[Minimum business continuity objective for each business function]\n\n## Resource Requirements\n[Resources required to recover each business function]\n\n## Interim Recovery Strategies\n[Interim recovery strategies for each business function]\n\n## Long-Term Recovery Strategies\n[Long-term recovery strategies for each business function]\n\n## Plan Testing and Maintenance\n[How the business impact analysis is tested and maintained]\n\n## Review and Approval\n[How the business impact analysis is reviewed and approved]""",
    
    "Data Classification Policy": """# Data Classification Policy Template\n\n## Overview\n[Statement of commitment to data classification]\n\n## Scope\n[What data and systems this policy covers]\n\n## Data Classification Levels\n### 1. Public Data\n- Description: [Data that can be freely shared]\n- Handling Requirements: [How to handle this data]\n- Examples: [Examples of public data]\n\n### 2. Internal Data\n- Description: [Data for internal use only]\n- Handling Requirements: [How to handle this data]\n- Examples: [Examples of internal data]\n\n### 3. Confidential Data\n- Description: [Sensitive data requiring protection]\n- Handling Requirements: [How to handle this data]\n- Examples: [Examples of confidential data]\n\n### 4. Restricted Data\n- Description: [Highly sensitive data with strict controls]\n- Handling Requirements: [How to handle this data]\n- Examples: [Examples of restricted data]\n\n## Data Ownership\n- Data Owners: [Who owns different types of data]\n- Responsibilities: [What data owners are responsible for]\n\n## Data Handling Procedures\n- Storage: [How to store data at each classification level]\n- Transmission: [How to transmit data at each classification level]\n- Disposal: [How to dispose of data at each classification level]\n- Access Control: [Who can access data at each classification level]\n\n## Training and Awareness\n[Requirements for data classification training]\n\n## Compliance and Enforcement\n[How compliance is measured and enforced]\n\n## Review and Updates\n[How often the policy is reviewed]""",
    
    "Incident Response Communication Plan": """# Incident Response Communication Plan Template\n\n## Overview\n[Brief description of the communication plan's purpose]\n\n## Objectives\n[What the communication plan aims to achieve]\n\n## Communication Team\n- Team Lead: [Name and contact information]\n- Spokesperson: [Name and contact information]\n- Technical Lead: [Name and contact information]\n- Legal Advisor: [Name and contact information]\n- HR Representative: [Name and contact information]\n\n## Stakeholder Groups\n### 1. Internal Stakeholders\n- Employees: [How to communicate with employees]\n- Management: [How to communicate with management]\n- IT Staff: [How to communicate with IT staff]\n\n### 2. External Stakeholders\n- Customers: [How to communicate with customers]\n- Vendors: [How to communicate with vendors]\n- Regulators: [How to communicate with regulators]\n- Media: [How to communicate with media]\n- Law Enforcement: [How to communicate with law enforcement]\n\n## Communication Channels\n- Email: [When and how to use email]\n- Phone: [When and how to use phone]\n- Website: [When and how to update website]\n- Social Media: [When and how to use social media]\n- Press Releases: [When and how to issue press releases]\n\n## Communication Templates\n### Initial Notification\n[Template for initial incident notification]\n\n### Progress Updates\n[Template for progress updates]\n\n### Resolution Notification\n[Template for resolution notification]\n\n### Post-Incident Report\n[Template for post-incident report]\n\n## Escalation Procedures\n[When and how to escalate communications]\n\n## Approval Process\n[Who must approve external communications]\n\n## Training and Awareness\n[Requirements for communication team training]""",
    
    "Vulnerability Management Process": """# Vulnerability Management Process Template\n\n## Overview\n[Brief description of the vulnerability management process]\n\n## Scope\n[What systems, applications, and networks this process covers]\n\n## Roles and Responsibilities\n- Vulnerability Manager: [Responsibilities]\n- System Owners: [Responsibilities]\n- Security Team: [Responsibilities]\n- IT Operations: [Responsibilities]\n\n## Vulnerability Management Lifecycle\n### 1. Discovery\n- Scanning Tools: [Tools used for vulnerability scanning]\n- Scanning Schedule: [How often scans are performed]\n- Asset Inventory: [How assets are tracked]\n\n### 2. Assessment\n- Risk Rating: [How vulnerabilities are rated]\n- Impact Analysis: [How impact is assessed]\n- Prioritization: [How vulnerabilities are prioritized]\n
### 3. Remediation\n- Patch Management: [How patches are applied]\n- Workarounds: [Temporary solutions for unpatched vulnerabilities]\n- Exception Process: [When and how to request exceptions]\n
### 4. Verification\n- Re-scanning: [How to verify remediation]\n- Penetration Testing: [When and how penetration testing is performed]\n- Compliance Checking: [How compliance is verified]\n
## Tools and Technologies\n[List of tools used in the vulnerability management process]\n
## Reporting\n- Dashboard: [What metrics are tracked]\n- Executive Reports: [What information is provided to executives]\n- Compliance Reports: [What information is provided for compliance]\n
## Training and Awareness\n[Requirements for vulnerability management training]\n
## Plan Review and Improvement\n[How the process is reviewed and improved]"""
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
            "downloads": 12500
        },
        "ISO 27001 Information Security Management System": {
            "description": "Template for implementing an ISO 27001 compliant ISMS",
            "category": "Security",
            "complexity": "Advanced",
            "compliance": ["ISO 27001"],
            "tags": ["information security", "ISMS", "compliance"],
            "author": "ISO",
            "rating": 4.7,
            "downloads": 9800
        },
        "OWASP Top 10 Mitigation Strategies": {
            "description": "Practical strategies for mitigating the OWASP Top 10 web application risks",
            "category": "Security",
            "complexity": "Intermediate",
            "compliance": ["OWASP"],
            "tags": ["web security", "OWASP", "application security"],
            "author": "OWASP Community",
            "rating": 4.9,
            "downloads": 15200
        }
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
            "downloads": 8700
        },
        "HIPAA Security Rule Compliance Checklist": {
            "description": "Comprehensive checklist for HIPAA Security Rule compliance",
            "category": "Compliance",
            "complexity": "Intermediate",
            "compliance": ["HIPAA"],
            "tags": ["healthcare", "HIPAA", "security", "compliance"],
            "author": "HHS OCR",
            "rating": 4.5,
            "downloads": 7600
        },
        "SOX IT General Controls Framework": {
            "description": "Framework for implementing SOX-compliant IT general controls",
            "category": "Compliance",
            "complexity": "Advanced",
            "compliance": ["SOX"],
            "tags": ["financial", "SOX", "ITGC", "controls"],
            "author": "SEC Compliance Team",
            "rating": 4.4,
            "downloads": 6500
        }
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
            "downloads": 11200
        },
        "CI/CD Pipeline Security Checklist": {
            "description": "Security checklist for securing CI/CD pipelines",
            "category": "DevOps",
            "complexity": "Intermediate",
            "compliance": [],
            "tags": ["CI/CD", "pipeline security", "DevSecOps"],
            "author": "DevSecOps Community",
            "rating": 4.7,
            "downloads": 9800
        },
        "Infrastructure as Code Security Guide": {
            "description": "Guide for securing infrastructure deployed through IaC tools",
            "category": "DevOps",
            "complexity": "Intermediate",
            "compliance": [],
            "tags": ["IaC", "Terraform", "security", "cloud"],
            "author": "Cloud Security Alliance",
            "rating": 4.6,
            "downloads": 8900
        }
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
            "downloads": 7200
        },
        "Digital Transformation Roadmap": {
            "description": "Step-by-step roadmap for enterprise digital transformation",
            "category": "Business",
            "complexity": "Advanced",
            "compliance": [],
            "tags": ["digital transformation", "change management", "strategy"],
            "author": "Digital Transformation Experts",
            "rating": 4.7,
            "downloads": 8400
        },
        "Vendor Risk Management Framework": {
            "description": "Framework for assessing and managing third-party vendor risks",
            "category": "Business",
            "complexity": "Advanced",
            "compliance": [],
            "tags": ["vendor management", "third-party risk", "supply chain"],
            "author": "Risk Management Association",
            "rating": 4.6,
            "downloads": 6800
        }
    }
}

# Adversarial Testing Presets
ADVERSARIAL_PRESETS = {
    "Security Hardening": {
        "name": "🔐 Security Hardening",
        "description": "Focus on identifying and closing security gaps, enforcing least privilege, and adding comprehensive error handling.",
        "red_team_models": ["openai/gpt-4o-mini", "anthropic/claude-3-haiku", "google/gemini-1.5-flash"],
        "blue_team_models": ["openai/gpt-4o", "anthropic/claude-3-sonnet", "google/gemini-1.5-pro"],
        "min_iter": 5,
        "max_iter": 15,
        "confidence_threshold": 95,
        "review_type": "General SOP",
        "compliance_requirements": "Security best practices, OWASP guidelines, least privilege principle",
        "advanced_settings": {
            "critique_depth": 8,
            "patch_quality": 9,
            "detailed_tracking": True,
            "early_stopping": True
        }
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
            "performance_analytics": True
        }
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
            "target_complexity": 50
        }
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
            "target_length": 500
        }
    },
    "Code Review": {
        "name": "💻 Code Review",
        "description": "Specialized testing for software development protocols and code reviews.",
        "red_team_models": ["openai/gpt-4o", "anthropic/claude-3-opus", "codellama/codellama-70b-instruct"],
        "blue_team_models": ["openai/gpt-4o", "anthropic/claude-3-sonnet", "codellama/codellama-70b-instruct"],
        "min_iter": 3,
        "max_iter": 10,
        "confidence_threshold": 90,
        "review_type": "Code Review",
        "compliance_requirements": "Clean code principles, security best practices, performance optimization",
        "advanced_settings": {
            "critique_depth": 9,
            "patch_quality": 9,
            "detailed_tracking": True,
            "performance_analytics": True
        }
    },
    "Mission Critical": {
        "name": "🔥 Mission Critical",
        "description": "Maximum rigor for high-stakes protocols requiring the highest assurance.",
        "red_team_models": ["openai/gpt-4o", "anthropic/claude-3-opus", "google/gemini-1.5-pro"],
        "blue_team_models": ["openai/gpt-4o", "anthropic/claude-3-sonnet", "google/gemini-1.5-pro"],
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
            "early_stopping": False
        }
    },
    "AI Safety Review": {
        "name": "🛡️ AI Safety Review",
        "description": "Specialized testing for AI safety considerations, bias detection, and ethical alignment.",
        "red_team_models": ["openai/gpt-4o", "anthropic/claude-3-opus", "google/gemini-1.5-pro"],
        "blue_team_models": ["openai/gpt-4o", "anthropic/claude-3-sonnet", "meta-llama/llama-3-70b-instruct"],
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
            "explainability_focus": True
        }
    },
    "Privacy Protection": {
        "name": "🔒 Privacy Protection",
        "description": "Focus on data privacy protection, consent mechanisms, and regulatory compliance.",
        "red_team_models": ["openai/gpt-4o-mini", "anthropic/claude-3-haiku", "google/gemini-1.5-flash"],
        "blue_team_models": ["openai/gpt-4o", "anthropic/claude-3-sonnet", "google/gemini-1.5-pro"],
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
            "consent_mechanisms": True
        }
    }
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

def save_config_profile(profile_name: str) -> bool:
    """Save current configuration as a profile.
    
    Args:
        profile_name (str): Name for the profile
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        profile_data = {
            "provider": st.session_state.provider,
            "base_url": st.session_state.base_url,
            "model": st.session_state.model,
            "temperature": st.session_state.temperature,
            "top_p": st.session_state.top_p,
            "frequency_penalty": st.session_state.frequency_penalty,
            "presence_penalty": st.session_state.presence_penalty,
            "max_tokens": st.session_state.max_tokens,
            "max_iterations": st.session_state.max_iterations,
            "system_prompt": st.session_state.system_prompt,
            "adversarial_confidence": st.session_state.adversarial_confidence,
            "adversarial_min_iter": st.session_state.adversarial_min_iter,
            "adversarial_max_iter": st.session_state.adversarial_max_iter,
            "adversarial_max_tokens": st.session_state.adversarial_max_tokens,
        }
        
        # Save to session state
        if "config_profiles" not in st.session_state:
            st.session_state.config_profiles = {}
        st.session_state.config_profiles[profile_name] = profile_data
        return True
    except Exception as e:
        st.error(f"Error saving profile: {e}")
        return False

def load_config_profile(profile_name: str) -> bool:
    """Load a configuration profile.
    
    Args:
        profile_name (str): Name of the profile to load
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        if "config_profiles" not in st.session_state:
            st.session_state.config_profiles = {}
            
        if profile_name not in st.session_state.config_profiles:
            # Check if it's a built-in profile
            if profile_name in CONFIG_PROFILES:
                profile_data = CONFIG_PROFILES[profile_name]
            else:
                st.error(f"Profile '{profile_name}' not found.")
                return False
        else:
            profile_data = st.session_state.config_profiles[profile_name]
            
        # Apply profile data to session state
        for key, value in profile_data.items():
            if key in st.session_state:
                st.session_state[key] = value
                
        return True
    except Exception as e:
        st.error(f"Error loading profile: {e}")
        return False

def list_config_profiles() -> List[str]:
    """List all available configuration profiles.
    
    Returns:
        List[str]: List of profile names
    """
    profiles = list(CONFIG_PROFILES.keys())
    if "config_profiles" in st.session_state:
        profiles.extend(list(st.session_state.config_profiles.keys()))
    return sorted(list(set(profiles)))

def list_protocol_templates() -> List[str]:
    """List all available protocol templates.
    
    Returns:
        List[str]: List of template names
    """
    return list(PROTOCOL_TEMPLATES.keys())

def load_protocol_template(template_name: str) -> str:
    """Load a protocol template.
    
    Args:
        template_name (str): Name of the template to load
        
    Returns:
        str: Template content
    """
    return PROTOCOL_TEMPLATES.get(template_name, "")

def list_adversarial_presets() -> List[str]:
    """List all available adversarial testing presets.
    
    Returns:
        List[str]: List of preset names
    """
    return list(ADVERSARIAL_PRESETS.keys())

def load_adversarial_preset(preset_name: str) -> Dict:
    """Load an adversarial testing preset.
    
    Args:
        preset_name (str): Name of the preset to load
        
    Returns:
        Dict: Preset configuration
    """
    return ADVERSARIAL_PRESETS.get(preset_name, {})

def apply_adversarial_preset(preset_name: str) -> bool:
    """Apply an adversarial testing preset to the current session state.
    
    Args:
        preset_name (str): Name of the preset to apply
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        preset = load_adversarial_preset(preset_name)
        if not preset:
            return False
            
        # Apply preset configuration to session state
        st.session_state.red_team_models = preset.get("red_team_models", [])
        st.session_state.blue_team_models = preset.get("blue_team_models", [])
        st.session_state.adversarial_min_iter = preset.get("min_iter", 3)
        st.session_state.adversarial_max_iter = preset.get("max_iter", 10)
        st.session_state.adversarial_confidence = preset.get("confidence_threshold", 85)
        st.session_state.adversarial_review_type = preset.get("review_type", "General SOP")
        st.session_state.compliance_requirements = preset.get("compliance_requirements", "")
        
        # Apply advanced settings if present
        advanced_settings = preset.get("advanced_settings", {})
        if advanced_settings:
            st.session_state.adversarial_critique_depth = advanced_settings.get("critique_depth", 5)
            st.session_state.adversarial_patch_quality = advanced_settings.get("patch_quality", 5)
            st.session_state.adversarial_detailed_tracking = advanced_settings.get("detailed_tracking", False)
            st.session_state.adversarial_performance_analytics = advanced_settings.get("performance_analytics", False)
            st.session_state.adversarial_early_stopping = advanced_settings.get("early_stopping", False)
            if "target_complexity" in advanced_settings:
                st.session_state.adversarial_target_complexity = advanced_settings.get("target_complexity", 0)
            if "target_length" in advanced_settings:
                st.session_state.adversarial_target_length = advanced_settings.get("target_length", 0)
            
        return True
    except Exception as e:
        st.error(f"Error applying preset: {e}")
        return False

# ------------------------------------------------------------------
# Version Control and Collaboration Functions
# ------------------------------------------------------------------

def create_new_version(protocol_text: str, version_name: str = "", comment: str = "") -> str:
    """Create a new version of the protocol.
    
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
        "name": version_name or f"Version {len(st.session_state.protocol_versions) + 1}",
        "timestamp": timestamp,
        "protocol_text": protocol_text,
        "comment": comment,
        "author": "Current User",  # In a real implementation, this would be the actual user
        "complexity_metrics": calculate_protocol_complexity(protocol_text),
        "structure_analysis": extract_protocol_structure(protocol_text)
    }
    
    with st.session_state.thread_lock:
        st.session_state.protocol_versions.append(version)
        st.session_state.current_version_id = version_id
    
    return version_id

def load_version(version_id: str) -> bool:
    """Load a specific version of the protocol.
    
    Args:
        version_id (str): ID of the version to load
        
    Returns:
        bool: True if successful, False otherwise
    """
    with st.session_state.thread_lock:
        for version in st.session_state.protocol_versions:
            if version["id"] == version_id:
                st.session_state.protocol_text = version["protocol_text"]
                st.session_state.current_version_id = version_id
                return True
    return False

def get_version_history() -> List[Dict]:
    """Get the version history.
    
    Returns:
        List[Dict]: List of versions
    """
    with st.session_state.thread_lock:
        return st.session_state.protocol_versions.copy()

def add_comment(comment_text: str, version_id: str = None) -> str:
    """Add a comment to a version or the current protocol.
    
    Args:
        comment_text (str): The comment text
        version_id (str): Optional version ID to comment on
        
    Returns:
        str: Comment ID
    """
    comment_id = str(uuid.uuid4())
    timestamp = datetime.now().isoformat()
    
    comment = {
        "id": comment_id,
        "text": comment_text,
        "timestamp": timestamp,
        "author": "Current User",  # In a real implementation, this would be the actual user
        "version_id": version_id or st.session_state.current_version_id
    }
    
    with st.session_state.thread_lock:
        st.session_state.comments.append(comment)
    
    return comment_id

def get_comments(version_id: str = None) -> List[Dict]:
    """Get comments for a specific version or all comments.
    
    Args:
        version_id (str): Optional version ID to get comments for
        
    Returns:
        List[Dict]: List of comments
    """
    with st.session_state.thread_lock:
        if version_id:
            return [c for c in st.session_state.comments if c["version_id"] == version_id]
        return st.session_state.comments.copy()

def export_project() -> Dict:
    """Export the entire project including versions and comments.
    
    Returns:
        Dict: Project data
    """
    with st.session_state.thread_lock:
        return {
            "project_name": st.session_state.project_name,
            "project_description": st.session_state.project_description,
            "versions": st.session_state.protocol_versions,
            "comments": st.session_state.comments,
            "collaborators": st.session_state.collaborators,
            "tags": st.session_state.tags,
            "export_timestamp": datetime.now().isoformat()
        }

def export_project_detailed() -> Dict:
    """Export the entire project with detailed analytics and history.
    
    Returns:
        Dict: Detailed project data
    """
    with st.session_state.thread_lock:
        # Get analytics if adversarial testing was run
        analytics = {}
        if st.session_state.adversarial_results:
            analytics = generate_advanced_analytics(st.session_state.adversarial_results)
        
        return {
            "project_name": st.session_state.project_name,
            "project_description": st.session_state.project_description,
            "versions": st.session_state.protocol_versions,
            "comments": st.session_state.comments,
            "collaborators": st.session_state.collaborators,
            "tags": st.session_state.tags,
            "export_timestamp": datetime.now().isoformat(),
            "analytics": analytics,
            "adversarial_results": st.session_state.adversarial_results,
            "evolution_history": st.session_state.evolution_log,
            "model_performance": st.session_state.adversarial_model_performance
        }


def generate_shareable_link(project_data: Dict) -> str:
    """Generate a shareable link for the project.
    
    Args:
        project_data (Dict): Project data to share
        
    Returns:
        str: Shareable link
    """
    # In a real implementation, this would generate a real shareable link
    # For now, we'll simulate it
    project_id = hashlib.md5(json.dumps(project_data, sort_keys=True).encode()).hexdigest()[:16]
    return f"https://open-evolve.app/shared/{project_id}"


def get_version_by_name(version_name: str) -> Optional[Dict]:
    """Get a specific version by name.
    
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


def get_version_by_timestamp(timestamp: str) -> Optional[Dict]:
    """Get a specific version by timestamp.
    
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

def delete_version(version_id: str) -> bool:
    """Delete a specific version.
    
    Args:
        version_id (str): ID of the version to delete
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        with st.session_state.thread_lock:
            st.session_state.protocol_versions = [
                v for v in st.session_state.protocol_versions if v["id"] != version_id
            ]
            # If we deleted the current version, set to the latest remaining version
            if st.session_state.current_version_id == version_id and st.session_state.protocol_versions:
                latest_version = st.session_state.protocol_versions[-1]
                st.session_state.protocol_text = latest_version["protocol_text"]
                st.session_state.current_version_id = latest_version["id"]
        return True
    except Exception as e:
        st.error(f"Error deleting version: {e}")
        return False

def branch_version(version_id: str, new_version_name: str) -> Optional[str]:
    """Create a new branch from an existing version.
    
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
    new_version_id = create_new_version(
        version["protocol_text"], 
        new_version_name, 
        f"Branched from {version['name']}"
    )
    
    # Add branch metadata
    with st.session_state.thread_lock:
        for v in st.session_state.protocol_versions:
            if v["id"] == new_version_id:
                v["branch_from"] = version_id
                v["branch_name"] = new_version_name
                break
    
    return new_version_id


# Collaborative Editing Functions
def initialize_collaborative_session(user_id: str, document_id: str) -> Dict:
    """Initialize a collaborative editing session.
    
    Args:
        user_id (str): ID of the user initiating the session
        document_id (str): ID of the document to edit
        
    Returns:
        Dict: Session information
    """
    session_id = str(uuid.uuid4())
    timestamp = datetime.now().isoformat()
    
    session_info = {
        "session_id": session_id,
        "document_id": document_id,
        "created_by": user_id,
        "created_at": timestamp,
        "participants": [user_id],
        "document_snapshot": st.session_state.protocol_text,
        "edit_operations": [],
        "conflict_resolutions": [],
        "session_status": "active"
    }
    
    # Store session in state
    if "collaborative_sessions" not in st.session_state:
        st.session_state.collaborative_sessions = {}
    st.session_state.collaborative_sessions[session_id] = session_info
    
    return session_info

def join_collaborative_session(session_id: str, user_id: str) -> bool:
    """Join an existing collaborative editing session.
    
    Args:
        session_id (str): ID of the session to join
        user_id (str): ID of the user joining
        
    Returns:
        bool: True if successful, False otherwise
    """
    if "collaborative_sessions" not in st.session_state:
        return False
    
    if session_id not in st.session_state.collaborative_sessions:
        return False
    
    session = st.session_state.collaborative_sessions[session_id]
    if user_id not in session["participants"]:
        session["participants"].append(user_id)
        
        # Notify other participants
        session["edit_operations"].append({
            "type": "user_joined",
            "user_id": user_id,
            "timestamp": datetime.now().isoformat()
        })
    
    return True

def leave_collaborative_session(session_id: str, user_id: str) -> bool:
    """Leave a collaborative editing session.
    
    Args:
        session_id (str): ID of the session to leave
        user_id (str): ID of the user leaving
        
    Returns:
        bool: True if successful, False otherwise
    """
    if "collaborative_sessions" not in st.session_state:
        return False
    
    if session_id not in st.session_state.collaborative_sessions:
        return False
    
    session = st.session_state.collaborative_sessions[session_id]
    if user_id in session["participants"]:
        session["participants"].remove(user_id)
        
        # Notify other participants
        session["edit_operations"].append({
            "type": "user_left",
            "user_id": user_id,
            "timestamp": datetime.now().isoformat()
        })
    
    return True

def apply_edit_operation(session_id: str, user_id: str, operation: Dict) -> Dict:
    """Apply an edit operation in a collaborative session.
    
    Args:
        session_id (str): ID of the session
        user_id (str): ID of the user making the edit
        operation (Dict): Edit operation details
        
    Returns:
        Dict: Result of the operation including any conflicts
    """
    if "collaborative_sessions" not in st.session_state:
        return {"success": False, "error": "No collaborative sessions exist"}
    
    if session_id not in st.session_state.collaborative_sessions:
        return {"success": False, "error": "Session not found"}
    
    session = st.session_state.collaborative_sessions[session_id]
    
    # Add timestamp to operation
    operation["timestamp"] = datetime.now().isoformat()
    operation["user_id"] = user_id
    
    # Check for conflicts
    conflict_result = detect_conflicts(session, operation)
    
    if conflict_result["has_conflict"]:
        # Record conflict
        conflict_record = {
            "conflict_id": str(uuid.uuid4()),
            "operation": operation,
            "conflicting_operations": conflict_result["conflicting_operations"],
            "detected_at": datetime.now().isoformat(),
            "resolution_status": "pending"
        }
        
        session["conflict_resolutions"] = session.get("conflict_resolutions", []) # Ensure it's a list
        session["conflict_resolutions"] = session["conflict_resolutions"] + [conflict_record] # Append
        
        return {
            "success": True,
            "conflict_detected": True,
            "conflict_record": conflict_record,
            "message": "Conflict detected. Please resolve before continuing."
        }
    else:
        # Apply operation
        session["edit_operations"] = session.get("edit_operations", []) # Ensure it's a list
        session["edit_operations"] = session["edit_operations"] + [operation] # Append
        
        return {
            "success": True,
            "conflict_detected": False,
            "message": "Operation applied successfully."
        }

def detect_conflicts(session: Dict, new_operation: Dict) -> Dict:
    """Detect conflicts between a new operation and existing operations.
    
    Args:
        session (Dict): Collaborative session data
        new_operation (Dict): New operation to check for conflicts
        
    Returns:
        Dict: Conflict detection results
    """
    conflicting_operations = []
    
    # For simplicity, we'll check if there are overlapping edits in the same region
    new_start = new_operation.get("start_pos", 0)
    new_end = new_operation.get("end_pos", 0)
    
    for existing_op in session["edit_operations"][-10:]:  # Check last 10 operations
        if existing_op.get("user_id") != new_operation.get("user_id"):
            existing_start = existing_op.get("start_pos", 0)
            existing_end = existing_op.get("end_pos", 0)
            
            # Check for overlap
            if (new_start < existing_end and new_end > existing_start):
                conflicting_operations.append(existing_op)
    
    return {
        "has_conflict": len(conflicting_operations) > 0,
        "conflicting_operations": conflicting_operations
    }

def resolve_conflict(session_id: str, conflict_id: str, resolution: str) -> bool:
    """Resolve a conflict in a collaborative session.
    
    Args:
        session_id (str): ID of the session
        conflict_id (str): ID of the conflict to resolve
        resolution (str): Resolution strategy ('accept_new', 'accept_existing', 'merge')
        
    Returns:
        bool: True if successful, False otherwise
    """
    if "collaborative_sessions" not in st.session_state:
        return False
    
    if session_id not in st.session_state.collaborative_sessions:
        return False
    
    session = st.session_state.collaborative_sessions[session_id]
    
    # Find conflict record
    conflict_record = None
    for conflict in session["conflict_resolutions"]:
        if conflict["conflict_id"] == conflict_id:
            conflict_record = conflict
            break
    
    if not conflict_record:
        return False
    
    # Apply resolution
    conflict_record["resolution"] = resolution
    conflict_record["resolved_at"] = datetime.now().isoformat()
    conflict_record["resolution_status"] = "resolved"
    
    return True

def get_session_state(session_id: str) -> Optional[Dict]:
    """Get the current state of a collaborative session.
    
    Args:
        session_id (str): ID of the session
        
    Returns:
        Optional[Dict]: Session state or None if not found
    """
    if "collaborative_sessions" not in st.session_state:
        return None
    
    return st.session_state.collaborative_sessions.get(session_id)

def synchronize_document(session_id: str) -> Dict:
    """Synchronize document state across all participants.
    
    Args:
        session_id (str): ID of the session to synchronize
        
    Returns:
        Dict: Synchronization result
    """
    if "collaborative_sessions" not in st.session_state:
        return {"success": False, "error": "No collaborative sessions exist"}
    
    if session_id not in st.session_state.collaborative_sessions:
        return {"success": False, "error": "Session not found"}
    
    session = st.session_state.collaborative_sessions[session_id]
    
    # Reconstruct document from operations
    document_text = session["document_snapshot"]
    
    # Apply all operations in order
    for operation in sorted(session["edit_operations"], key=lambda x: x["timestamp"]):
        if operation["type"] == "insert":
            document_text = (
                document_text[:operation["start_pos"]] + 
                operation["text"] + 
                document_text[operation["start_pos"]:]
            )
        elif operation["type"] == "delete":
            document_text = (
                document_text[:operation["start_pos"]] + 
                document_text[operation["end_pos"]:]
            )
        elif operation["type"] == "replace":
            document_text = (
                document_text[:operation["start_pos"]] + 
                operation["text"] + 
                document_text[operation["end_pos"]:]
            )
    
    return {
        "success": True,
        "document_text": document_text,
        "participant_count": len(session["participants"]),
        "operation_count": len(session["edit_operations"])
    }

def render_collaborative_editor_ui(session_id: str) -> str:
    """Render the collaborative editor UI.
    
    Args:
        session_id (str): ID of the collaborative session
        
    Returns:
        str: HTML formatted editor UI
    """
    session = get_session_state(session_id)
    if not session:
        return "<p>Session not found.</p>"
    
    # Get synchronized document
    sync_result = synchronize_document(session_id)
    document_text = sync_result.get("document_text", "")
    
    html = f"""
    <div style="background-color: #f8f9fa; padding: 20px; border-radius: 10px; margin-bottom: 20px;">
        <h2 style="color: #4a6fa5; margin-top: 0; text-align: center;">👥 Collaborative Editor</h2>
        
        <!-- Session Info -->
        <div style="background-color: white; padding: 15px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin-bottom: 20px;">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <h3 style="margin: 0; color: #4a6fa5;">Session: {session_id[:8]}</h3>
                    <p style="margin: 5px 0 0 0; color: #666;">
                        <span style="background-color: #e8f5e9; color: #2e7d32; padding: 3px 8px; border-radius: 10px; font-size: 0.8em;">
                            {sync_result.get('participant_count', 0)} participants
                        </span>
                        <span style="background-color: #fff8e1; color: #f57f17; padding: 3px 8px; border-radius: 10px; font-size: 0.8em; margin-left: 10px;">
                            {sync_result.get('operation_count', 0)} edits
                        </span>
                    </p>
                </div>
                <div>
                    <button onclick="leaveSession('{session_id}')" style="background-color: #f44336; color: white; border: none; padding: 8px 15px; border-radius: 5px; cursor: pointer;">
                        Leave Session
                    </button>
                </div>
            </div>
        </div>
        
        <!-- Participant List -->
        <div style="background-color: white; padding: 15px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin-bottom: 20px;">
            <h3 style="color: #4a6fa5; margin-top: 0;">Participants</h3>
            <div style="display: flex; gap: 10px; flex-wrap: wrap;">
    """
    
    for participant in session.get("participants", []):
        html += f"""
        <div style="background-color: #e3f2fd; color: #1565c0; padding: 8px 15px; border-radius: 20px; font-size: 0.9em;">
            👤 {participant[:8]}
        </div>
        """
    
    html += """
            </div>
        </div>
        
        <!-- Document Editor -->
        <div style="background-color: white; padding: 15px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin-bottom: 20px;">
            <h3 style="color: #4a6fa5; margin-top: 0;">Document Editor</h3>
            <textarea id="collaborativeEditor" style="width: 100%; height: 400px; padding: 10px; border: 1px solid #ddd; border-radius: 5px; font-family: monospace;">{document_text}</textarea>
            <div style="margin-top: 10px; display: flex; gap: 10px;">
                <button onclick="saveChanges()" style="background-color: #4a6fa5; color: white; border: none; padding: 8px 15px; border-radius: 5px; cursor: pointer;">
                    Save Changes
                </button>
                <button onclick="refreshView()" style="background-color: #6b8cbc; color: white; border: none; padding: 8px 15px; border-radius: 5px; cursor: pointer;">
                    Refresh
                </button>
            </div>
        </div>
        
        <!-- Conflict Resolution Panel -->
        <div id="conflictPanel" style="background-color: #fff3e0; padding: 15px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); display: none;">
            <h3 style="color: #ef6c00; margin-top: 0;">⚠️ Conflict Detected</h3>
            <p id="conflictMessage">Resolving edit conflicts...</p>
            <div style="display: flex; gap: 10px; margin-top: 10px;">
                <button onclick="acceptNew()" style="background-color: #4caf50; color: white; border: none; padding: 8px 15px; border-radius: 5px; cursor: pointer;">
                    Accept My Changes
                </button>
                <button onclick="acceptExisting()" style="background-color: #ff9800; color: white; border: none; padding: 8px 15px; border-radius: 5px; cursor: pointer;">
                    Accept Their Changes
                </button>
                <button onclick="mergeChanges()" style="background-color: #2196f3; color: white; border: none; padding: 8px 15px; border-radius: 5px; cursor: pointer;">
                    Merge Changes
                </button>
            </div>
        </div>
    </div>
    
    <script>
    let sessionId = '{session_id}';
    let editor = document.getElementById('collaborativeEditor');
    let conflictPanel = document.getElementById('conflictPanel');
    
    function leaveSession(sessionId) {
        // In a real implementation, this would leave the session
        alert('Leaving session: ' + sessionId);
    }
    
    function saveChanges() {
        // In a real implementation, this would save changes
        let text = editor.value;
        alert('Saving changes...');
    }
    
    function refreshView() {
        // In a real implementation, this would refresh the view
        alert('Refreshing view...');
    }
    
    function acceptNew() {
        conflictPanel.style.display = 'none';
        alert('Accepting your changes...');
    }
    
    function acceptExisting() {
        conflictPanel.style.display = 'none';
        alert('Accepting their changes...');
    }
    
    function mergeChanges() {
        conflictPanel.style.display = 'none';
        alert('Merging changes...');
    }
    
    // Simulate real-time updates
    setInterval(function() {
        // In a real implementation, this would fetch updates
        console.log('Checking for updates...');
    }, 5000);
    </script>
    """.format(document_text=document_text.replace('"', '&quot;'))
    
    return html

def validate_protocol(protocol_text: str, validation_type: str = "generic") -> Dict:
    """Validate a protocol against predefined rules.
    
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
            "suggestions": ["Please provide protocol text to validate"]
        }
    
    # Get validation rules
    rules = VALIDATION_RULES.get(validation_type, VALIDATION_RULES.get("generic", {}))
    
    errors = []
    warnings = []
    suggestions = []
    
    # Check length
    char_count = len(protocol_text)
    if "max_length" in rules and char_count > rules["max_length"]:
        errors.append(f"Protocol exceeds maximum length of {rules['max_length']} characters")
    
    # Check required sections
    if "required_sections" in rules:
        sections = extract_protocol_structure(protocol_text)["section_count"]
        missing_sections = []
        for section in rules["required_sections"]:
            if section.lower() not in protocol_text.lower():
                missing_sections.append(section)
        if missing_sections:
            errors.append(f"Missing required sections: {', '.join(missing_sections)}")
    
    # Check required keywords
    if "required_keywords" in rules:
        missing_keywords = []
        for keyword in rules["required_keywords"]:
            if keyword.lower() not in protocol_text.lower():
                missing_keywords.append(keyword)
        if missing_keywords:
            warnings.append(f"Consider adding these keywords: {', '.join(missing_keywords)}")
    
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
        suggestions.append(f"Increase protocol complexity (current: {complexity_score}, minimum: {rules['min_complexity']})")
    
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
        "complexity_metrics": complexity
    }

def render_validation_results(protocol_text: str, validation_type: str = "generic") -> str:
    """Render validation results in a formatted display.
    
    Args:
        protocol_text (str): Protocol text to validate
        validation_type (str): Type of validation to perform
        
    Returns:
        str: HTML formatted validation results
    """
    results = validate_protocol(protocol_text, validation_type)
    
    html = f"""
    <div style="background-color: #f8f9fa; padding: 20px; border-radius: 10px; margin-bottom: 20px;">
        <h2 style="color: #4a6fa5; margin-top: 0; text-align: center;">✅ Protocol Validation Results</h2>
        
        <!-- Overall Score -->
        <div style="display: flex; justify-content: center; align-items: center; margin-bottom: 20px;">
            <div style="background: {
                        'linear-gradient(135deg, #4caf50, #81c784)' if results['score'] >= 80 else 
                        'linear-gradient(135deg, #ff9800, #ffb74d)' if results['score'] >= 60 else 
                        'linear-gradient(135deg, #f44336, #e57373)'
                    }; 
                        color: white; border-radius: 50%; width: 120px; height: 120px; 
                        display: flex; justify-content: center; align-items: center; 
                        font-size: 2em; font-weight: bold;">
                {results['score']}%
            </div>
        </div>
        <p style="text-align: center; margin-top: 0; font-size: 1.2em; font-weight: bold;">
            {'Valid Protocol' if results['valid'] else 'Protocol Needs Improvement'}
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
        ("Unique Words", complexity["unique_words"])
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

def list_report_templates() -> List[str]:
    """List all available report templates.
    
    Returns:
        List[str]: List of report template names
    """
    return list(REPORT_TEMPLATES.keys())

def get_report_template_details(template_name: str) -> Optional[Dict]:
    """Get details for a specific report template.
    
    Args:
        template_name (str): Name of the template
        
    Returns:
        Optional[Dict]: Template details or None if not found
    """
    return REPORT_TEMPLATES.get(template_name)

def generate_custom_report(template_name: str, data: Dict) -> str:
    """Generate a custom report based on a template.
    
    Args:
        template_name (str): Name of the template to use
        data (Dict): Data to populate the report
        
    Returns:
        str: Generated report content
    """
    template = get_report_template_details(template_name)
    if not template:
        return f"# Error: Template '{template_name}' not found\n\nUnable to generate report."
    
    report_content = f"# {template['name']}\n\n"
    report_content += f"*Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*

"
    
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

def render_report_generator_ui() -> str:
    """Render the report generator UI.
    
    Returns:
        str: HTML formatted report generator UI
    """
    html = """
    <div style="background-color: #f8f9fa; padding: 20px; border-radius: 10px; margin-bottom: 20px;">
        <h2 style="color: #4a6fa5; margin-top: 0; text-align: center;">📋 Report Generator</h2>
        
        <!-- Template Selection -->
        <div style="background-color: white; padding: 15px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin-bottom: 20px;">
            <h3 style="color: #4a6fa5; margin-top: 0;">Select Report Template</h3>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 15px;">
    """
    
    # Add template cards
    for template_name, template in REPORT_TEMPLATES.items():
        html += f"""
        <div style="background-color: #e3f2fd; border-radius: 8px; padding: 15px; cursor: pointer;" 
             onclick="selectTemplate('{template_name}')">
            <h4 style="color: #1565c0; margin-top: 0;">{template['name']}</h4>
            <p style="color: #666; font-size: 0.9em; margin-bottom: 10px;">{template['description']}</p>
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <span style="background-color: #bbdefb; color: #0d47a1; padding: 3px 8px; border-radius: 10px; font-size: 0.8em;">
                    {template['format'].upper()}
                </span>
                <span style="font-size: 0.8em; color: #999;">
                    {len(template.get('sections', []))} sections
                </span>
            </div>
        </div>
        """
    
    html += """
            </div>
        </div>
        
        <!-- Report Configuration -->
        <div id="reportConfig" style="background-color: white; padding: 15px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin-bottom: 20px; display: none;">
            <h3 style="color: #4a6fa5; margin-top: 0;">Configure Report</h3>
            <div id="reportFields">
                <!-- Fields will be populated by JavaScript -->
            </div>
            <div style="margin-top: 20px;">
                <button onclick="generateReport()" style="background-color: #4a6fa5; color: white; border: none; padding: 10px 20px; border-radius: 5px; cursor: pointer; font-size: 1em;">
                    Generate Report
                </button>
            </div>
        </div>
        
        <!-- Generated Report Preview -->
        <div id="reportPreview" style="background-color: white; padding: 15px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); display: none;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px;">
                <h3 style="color: #4a6fa5; margin: 0;">Report Preview</h3>
                <div>
                    <button onclick="downloadReport('pdf')" style="background-color: #f44336; color: white; border: none; padding: 8px 15px; border-radius: 5px; cursor: pointer; margin-right: 10px;">
                        Download PDF
                    </button>
                    <button onclick="downloadReport('md')" style="background-color: #4caf50; color: white; border: none; padding: 8px 15px; border-radius: 5px; cursor: pointer;">
                        Download Markdown
                    </button>
                </div>
            </div>
            <div id="reportContent" style="border: 1px solid #ddd; border-radius: 5px; padding: 15px; background-color: #fafafa; font-family: monospace; white-space: pre-wrap; max-height: 400px; overflow-y: auto;">
                <!-- Report content will appear here -->
            </div>
        </div>
    </div>
    
    <script>
    let selectedTemplate = null;
    
    function selectTemplate(templateName) {
        selectedTemplate = templateName;
        document.getElementById('reportConfig').style.display = 'block';
        
        // Populate fields based on template
        const fieldsContainer = document.getElementById('reportFields');
        fieldsContainer.innerHTML = '';
        
        // In a real implementation, this would dynamically generate fields
        // based on the template structure
        fieldsContainer.innerHTML = `
            <div style="margin-bottom: 15px;">
                <label style="display: block; margin-bottom: 5px; font-weight: bold;">Report Title</label>
                <input type="text" id="reportTitle" style="width: 100%; padding: 8px; border: 1px solid #ddd; border-radius: 5px;" placeholder="Enter report title">
            </div>
            <div style="margin-bottom: 15px;">
                <label style="display: block; margin-bottom: 5px; font-weight: bold;">Prepared By</label>
                <input type="text" id="preparedBy" style="width: 100%; padding: 8px; border: 1px solid #ddd; border-radius: 5px;" placeholder="Enter preparer name">
            </div>
            <div style="margin-bottom: 15px;">
                <label style="display: block; margin-bottom: 5px; font-weight: bold;">Date Range</label>
                <input type="text" id="dateRange" style="width: 100%; padding: 8px; border: 1px solid #ddd; border-radius: 5px;" placeholder="Enter date range">
            </div>
        `;
    }
    
    function generateReport() {
        if (!selectedTemplate) {
            alert('Please select a template first.');
            return;
        }
        
        // In a real implementation, this would collect form data and generate the report
        const reportContent = `# ${selectedTemplate.replace('_', ' ').toUpperCase()} Report

*Generated on: ${new Date().toLocaleString()}*

## Executive Summary

This report provides a comprehensive analysis of the selected topic.

## Key Findings

1. First key finding
2. Second key finding
3. Third key finding

## Recommendations

- Recommendation one
- Recommendation two
- Recommendation three

## Conclusion

This concludes the report.`;
        
        document.getElementById('reportContent').textContent = reportContent;
        document.getElementById('reportPreview').style.display = 'block';
    }
    
    function downloadReport(format) {
        alert(`Downloading report in ${format.toUpperCase()} format...`);
        // In a real implementation, this would generate and download the actual file
    }
    </script>
    """
    
    return html


# External Integration Functions
EXTERNAL_INTEGRATIONS = {
    "github": {
        "name": "GitHub",
        "description": "Integrate with GitHub repositories",
        "auth_type": "token",
        "base_url": "https://api.github.com",
        "capabilities": ["pull_requests", "issues", "repositories"]
    },
    "gitlab": {
        "name": "GitLab",
        "description": "Integrate with GitLab repositories",
        "auth_type": "token",
        "base_url": "https://gitlab.com/api/v4",
        "capabilities": ["merge_requests", "issues", "projects"]
    },
    "jira": {
        "name": "Jira",
        "description": "Integrate with Jira issue tracking",
        "auth_type": "oauth",
        "base_url": "https://your-domain.atlassian.net/rest/api/3",
        "capabilities": ["issues", "projects", "sprints"]
    },
    "slack": {
        "name": "Slack",
        "description": "Integrate with Slack messaging",
        "auth_type": "bot_token",
        "base_url": "https://slack.com/api",
        "capabilities": ["messages", "channels", "users"]
    }
}

def get_version_timeline() -> List[Dict]:
    """Get a chronological timeline of all versions.
    
    Returns:
        List[Dict]: Sorted list of versions by timestamp
    """
    with st.session_state.thread_lock:
        versions = st.session_state.protocol_versions.copy()
    
    # Sort by timestamp
    versions.sort(key=lambda x: x["timestamp"])
    return versions

def render_version_timeline() -> str:
    """Render a visual timeline of versions.
    
    Returns:
        str: HTML formatted timeline
    """
    versions = get_version_timeline()
    
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
                    <h4 style="margin: 0; color: #4a6fa5;">{version['name']}</h4>
                    <span style="font-size: 0.9em; color: #666;">{timestamp}</span>
                </div>
                <p style="margin: 5px 0 0 0; color: #666;">{version.get('comment', 'No comment')}</p>
                <div style="margin-top: 10px; display: flex; gap: 10px;">
        """
        
        # Add action buttons
        html += f"""
                    <button onclick="loadVersion('{version['id']}')" style="background-color: #4a6fa5; color: white; border: none; padding: 5px 10px; border-radius: 4px; cursor: pointer; font-size: 0.8em;">Load</button>
        """
        
        if not is_current:
            html += f"""
                    <button onclick="branchVersion('{version['id']}', 'Branch of {version['name']}')" style="background-color: #6b8cbc; color: white; border: none; padding: 5px 10px; border-radius: 4px; cursor: pointer; font-size: 0.8em;">Branch</button>
            """
        
        html += """
                </div>
            </div>
        </div>
        """
    
    html += """
        </div>
    </div>
    <script>
    function loadVersion(versionId) {
        // In a real implementation, this would trigger a reload with the version
        alert('Loading version: ' + versionId);
    }
    
    function branchVersion(versionId, branchName) {
        // In a real implementation, this would create a new branch
        alert('Branching from version: ' + versionId + ' as ' + branchName);
    }
    </script>
    """
    
    return html


# External Integration Functions
EXTERNAL_INTEGRATIONS = {
    "github": {
        "name": "GitHub",
        "description": "Integrate with GitHub repositories",
        "auth_type": "token",
        "base_url": "https://api.github.com",
        "capabilities": ["pull_requests", "issues", "repositories"]
    },
    "gitlab": {
        "name": "GitLab",
        "description": "Integrate with GitLab repositories",
        "auth_type": "token",
        "base_url": "https://gitlab.com/api/v4",
        "capabilities": ["merge_requests", "issues", "projects"]
    },
    "jira": {
        "name": "Jira",
        "description": "Integrate with Jira issue tracking",
        "auth_type": "oauth",
        "base_url": "https://your-domain.atlassian.net/rest/api/3",
        "capabilities": ["issues", "projects", "sprints"]
    },
    "slack": {
        "name": "Slack",
        "description": "Integrate with Slack messaging",
        "auth_type": "bot_token",
        "base_url": "https://slack.com/api",
        "capabilities": ["messages", "channels", "users"]
    }
}

def export_protocol_as_template(protocol_text: str, template_name: str) -> Dict:
    """Export protocol as a reusable template.
    
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
        "tags": []
    }

def generate_ai_insights(protocol_text: str) -> Dict[str, Any]:
    """Generate AI-powered insights about the protocol.
    
    Args:
        protocol_text (str): Protocol text to analyze
        
    Returns:
        Dict[str, Any]: AI insights and recommendations
    """
    if not protocol_text:
        return {
            "overall_score": 0,
            "strengths": [],
            "weaknesses": [],
            "opportunities": [],
            "threats": [],
            "recommendations": [],
            "complexity_analysis": {},
            "readability_score": 0,
            "compliance_risk": "low"
        }
    
    # Calculate metrics
    complexity = calculate_protocol_complexity(protocol_text)
    structure = extract_protocol_structure(protocol_text)
    
    # Overall score calculation (weighted)
    structure_score = (
        (1 if structure["has_headers"] else 0) * 0.2 +
        (1 if structure["has_numbered_steps"] or structure["has_bullet_points"] else 0) * 0.2 +
        (1 if structure["has_preconditions"] else 0) * 0.15 +
        (1 if structure["has_postconditions"] else 0) * 0.15 +
        (1 if structure["has_error_handling"] else 0) * 0.15 +
        min(structure["section_count"] / 10, 1) * 0.15
    ) * 100
    
    complexity_score = max(0, 100 - complexity["complexity_score"])
    
    overall_score = (structure_score * 0.6 + complexity_score * 0.4)
    
    # Strengths
    strengths = []
    if structure["has_headers"]:
        strengths.append("✅ Well-structured with clear headers")
    if structure["has_numbered_steps"] or structure["has_bullet_points"]:
        strengths.append("✅ Uses lists or numbered steps for clarity")
    if structure["has_preconditions"]:
        strengths.append("✅ Defines clear preconditions")
    if structure["has_postconditions"]:
        strengths.append("✅ Specifies expected outcomes")
    if structure["has_error_handling"]:
        strengths.append("✅ Includes error handling procedures")
    if complexity["unique_words"] / max(1, complexity["word_count"]) > 0.6:
        strengths.append("✅ Good vocabulary diversity")
    
    # Weaknesses
    weaknesses = []
    if not structure["has_headers"]:
        weaknesses.append("❌ Lacks clear section headers")
    if not structure["has_numbered_steps"] and not structure["has_bullet_points"]:
        weaknesses.append("❌ Could use lists or numbered steps for better readability")
    if not structure["has_preconditions"]:
        weaknesses.append("❌ Missing preconditions specification")
    if not structure["has_postconditions"]:
        weaknesses.append("❌ No defined postconditions or expected outcomes")
    if not structure["has_error_handling"]:
        weaknesses.append("❌ Lacks error handling procedures")
    if complexity["avg_sentence_length"] > 25:
        weaknesses.append("❌ Sentences are quite long (hard to read)")
    if complexity["complexity_score"] > 60:
        weaknesses.append("❌ Protocol is quite complex")
    
    # Opportunities
    opportunities = []
    if complexity["word_count"] < 500:
        opportunities.append("✨ Protocol is brief - opportunity to add more detail")
    if structure["section_count"] == 0 and complexity["word_count"] > 300:
        opportunities.append("✨ Can improve organization with section headers")
    if not structure["has_preconditions"]:
        opportunities.append("✨ Add preconditions to clarify requirements")
    if not structure["has_postconditions"]:
        opportunities.append("✨ Define postconditions to specify expected outcomes")
    if not structure["has_error_handling"]:
        opportunities.append("✨ Include error handling for robustness")
    
    # Threats (potential issues)
    threats = []
    if complexity["complexity_score"] > 70:
        threats.append("⚠️ High complexity may lead to misinterpretation")
    if complexity["avg_sentence_length"] > 30:
        threats.append("⚠️ Long sentences may reduce clarity")
    if structure["section_count"] == 0 and complexity["word_count"] > 500:
        threats.append("⚠️ Lack of sections makes long protocols hard to navigate")
    
    # Recommendations
    recommendations = generate_protocol_recommendations(protocol_text)
    
    # Readability score
    readability_score = 100 - (complexity["avg_sentence_length"] / 50 * 100)
    readability_score = max(0, min(100, readability_score))
    
    # Compliance risk assessment
    compliance_risk = "low"
    if complexity["complexity_score"] > 70:
        compliance_risk = "high"
    elif complexity["complexity_score"] > 50:
        compliance_risk = "medium"
    
    return {
        "overall_score": round(overall_score, 1),
        "strengths": strengths,
        "weaknesses": weaknesses,
        "opportunities": opportunities,
        "threats": threats,
        "recommendations": recommendations,
        "complexity_analysis": complexity,
        "structure_analysis": structure,
        "readability_score": round(readability_score, 1),
        "compliance_risk": compliance_risk
    }

def render_ai_insights_dashboard(protocol_text: str) -> str:
    """Render an AI insights dashboard for the protocol.
    
    Args:
        protocol_text (str): Protocol text to analyze
        
    Returns:
        str: HTML formatted dashboard
    """
    insights = generate_ai_insights(protocol_text)
    
    # Create a visual dashboard
    html = """
    <div style="background: linear-gradient(135deg, #4a6fa5, #6b8cbc); color: white; padding: 20px; border-radius: 10px; margin-bottom: 20px;">
        <h2 style="margin-top: 0; text-align: center;">🤖 AI Insights Dashboard</h2>
        <div style="display: flex; justify-content: center; align-items: center;">
            <div style="background: white; color: #4a6fa5; border-radius: 50%; width: 100px; height: 100px; display: flex; justify-content: center; align-items: center; font-size: 2em; font-weight: bold;">
                """ + str(insights["overall_score"]) + """%
            </div>
        </div>
        <p style="text-align: center; margin-top: 10px;">Overall Protocol Quality Score</p>
    </div>
    """
    
    # Add metrics cards
    html += """
    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin-bottom: 20px;">
        <div style="background-color: #e8f5e9; padding: 15px; border-radius: 8px; border-left: 4px solid #4caf50;">
            <h4 style="margin-top: 0; color: #2e7d32;">📊 Readability</h4>
            <p style="font-size: 1.5em; font-weight: bold; margin: 0;">""" + str(insights["readability_score"]) + """%</p>
        </div>
        <div style="background-color: #fff8e1; padding: 15px; border-radius: 8px; border-left: 4px solid #ff9800;">
            <h4 style="margin-top: 0; color: #f57f17;">📋 Structure</h4>
            <p style="font-size: 1.5em; font-weight: bold; margin: 0;">""" + str(len([s for s in insights["structure_analysis"].values() if s])) + """/7</p>
        </div>
        <div style="background-color: #ffebee; padding: 15px; border-radius: 8px; border-left: 4px solid #f44336;">
            <h4 style="margin-top: 0; color: #c62828;">⚠️ Compliance Risk</h4>
            <p style="font-size: 1.5em; font-weight: bold; margin: 0; text-transform: capitalize;">""" + insights["compliance_risk"] + """</p>
        </div>
    </div>
    """
    
    # Add insights sections
    html += """
    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px;">
    """
    
    # Strengths
    if insights["strengths"]:
        html += """
        <div style="background-color: #e8f5e9; padding: 15px; border-radius: 8px;">
            <h3 style="color: #2e7d32; margin-top: 0;">✅ Strengths</h3>
            <ul style="padding-left: 20px;">
        """
        for strength in insights["strengths"][:5]:  # Limit to first 5
            html += f"<li>{strength}</li>"
        html += """
            </ul>
        </div>
        """
    
    # Weaknesses
    if insights["weaknesses"]:
        html += """
        <div style="background-color: #ffebee; padding: 15px; border-radius: 8px;">
            <h3 style="color: #c62828; margin-top: 0;">❌ Areas for Improvement</h3>
            <ul style="padding-left: 20px;">
        """
        for weakness in insights["weaknesses"][:5]:  # Limit to first 5
            html += f"<li>{weakness}</li>"
        html += """
            </ul>
        </div>
        """
    
    # Opportunities
    if insights["opportunities"]:
        html += """
        <div style="background-color: #e3f2fd; padding: 15px; border-radius: 8px;">
            <h3 style="color: #1565c0; margin-top: 0;">✨ Opportunities</h3>
            <ul style="padding-left: 20px;">
        """
        for opportunity in insights["opportunities"][:5]:  # Limit to first 5
            html += f"<li>{opportunity}</li>"
        html += """
            </ul>
        </div>
        """
    
    # Threats
    if insights["threats"]:
        html += """
        <div style="background-color: #fff3e0; padding: 15px; border-radius: 8px;">
            <h3 style="color: #ef6c00; margin-top: 0;">⚠️ Potential Threats</h3>
            <ul style="padding-left: 20px;">
        """
        for threat in insights["threats"][:5]:  # Limit to first 5
            html += f"<li>{threat}</li>"
        html += """
            </ul>
        </div>
        """
    
    html += "</div>"
    
    # Add recommendations
    if insights["recommendations"]:
        html += """
        <div style="background-color: #f3e5f5; padding: 15px; border-radius: 8px; margin-top: 20px;">
            <h3 style="color: #6a1b9a; margin-top: 0;">💡 AI Recommendations</h3>
            <ul style="padding-left: 20px;">
        """
        for recommendation in insights["recommendations"]:
            html += f"<li>{recommendation}</li>"
        html += """
            </ul>
        </div>
        """
    
    return html

def list_template_categories() -> List[str]:
    """List all template categories in the marketplace.
    
    Returns:
        List[str]: List of category names
    """
    return list(PROTOCOL_TEMPLATE_MARKETPLACE.keys())

def list_templates_in_category(category: str) -> List[str]:
    """List all templates in a specific category.
    
    Args:
        category (str): Category name
        
    Returns:
        List[str]: List of template names
    """
    if category in PROTOCOL_TEMPLATE_MARKETPLACE:
        return list(PROTOCOL_TEMPLATE_MARKETPLACE[category].keys())
    return []

def get_template_details(category: str, template_name: str) -> Optional[Dict]:
    """Get details for a specific template.
    
    Args:
        category (str): Category name
        template_name (str): Template name
        
    Returns:
        Optional[Dict]: Template details or None if not found
    """
    if category in PROTOCOL_TEMPLATE_MARKETPLACE:
        if template_name in PROTOCOL_TEMPLATE_MARKETPLACE[category]:
            return PROTOCOL_TEMPLATE_MARKETPLACE[category][template_name]
    return None

def search_templates(query: str) -> List[Tuple[str, str, Dict]]:
    """Search templates by query term.
    
    Args:
        query (str): Search query
        
    Returns:
        List[Tuple[str, str, Dict]]: List of (category, template_name, details) tuples
    """
    results = []
    query_lower = query.lower()
    
    for category, templates in PROTOCOL_TEMPLATE_MARKETPLACE.items():
        for template_name, details in templates.items():
            # Search in template name, description, tags
            if (query_lower in template_name.lower() or 
                query_lower in details.get("description", "").lower() or
                any(query_lower in tag.lower() for tag in details.get("tags", []))):
                results.append((category, template_name, details))
    
    # Sort by rating (descending)
    results.sort(key=lambda x: x[2].get("rating", 0), reverse=True)
    return results

def get_popular_templates(limit: int = 10) -> List[Tuple[str, str, Dict]]:
    """Get the most popular templates.
    
    Args:
        limit (int): Maximum number of templates to return
        
    Returns:
        List[Tuple[str, str, Dict]]: List of (category, template_name, details) tuples
    """
    all_templates = []
    
    for category, templates in PROTOCOL_TEMPLATE_MARKETPLACE.items():
        for template_name, details in templates.items():
            all_templates.append((category, template_name, details))
    
    # Sort by downloads (descending)
    all_templates.sort(key=lambda x: x[2].get("downloads", 0), reverse=True)
    return all_templates[:limit]

def get_top_rated_templates(limit: int = 10) -> List[Tuple[str, str, Dict]]:
    """Get the top-rated templates.
    
    Args:
        limit (int): Maximum number of templates to return
        
    Returns:
        List[Tuple[str, str, Dict]]: List of (category, template_name, details) tuples
    """
    all_templates = []
    
    for category, templates in PROTOCOL_TEMPLATE_MARKETPLACE.items():
        for template_name, details in templates.items():
            all_templates.append((category, template_name, details))
    
    # Sort by rating (descending)
    all_templates.sort(key=lambda x: x[2].get("rating", 0), reverse=True)
    return all_templates[:limit]

def render_template_marketplace_ui() -> str:
    """Render the template marketplace UI.
    
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
    popular_templates = get_popular_templates(6)
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
    def import_project(project_data: Dict) -> bool:
    """Import a project including versions and comments.
    
    Args:
        project_data (Dict): Project data to import
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        with st.session_state.thread_lock:
            st.session_state.project_name = project_data.get("project_name", "Imported Project")
            st.session_state.project_description = project_data.get("project_description", "")
            st.session_state.protocol_versions = project_data.get("versions", [])
            st.session_state.comments = project_data.get("comments", [])
            st.session_state.collaborators = project_data.get("collaborators", [])
            st.session_state.tags = project_data.get("tags", [])
            
            # Set current version to the latest one
            if st.session_state.protocol_versions:
                latest_version = st.session_state.protocol_versions[-1]
                st.session_state.protocol_text = latest_version["protocol_text"]
                st.session_state.current_version_id = latest_version["id"]
        return True
    except Exception as e:
        st.error(f"Error importing project: {e}")
        return False

def add_validation_rule(rule_name: str, rule_config: Dict) -> bool:
    """Add a new validation rule.
    
    Args:
        rule_name (str): Name of the rule
        rule_config (Dict): Configuration for the rule
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        VALIDATION_RULES[rule_name] = rule_config
        return True
    except Exception as e:
        st.error(f"Error adding validation rule: {e}")
        return False

def update_validation_rule(rule_name: str, rule_config: Dict) -> bool:
    """Update an existing validation rule.
    
    Args:
        rule_name (str): Name of the rule to update
        rule_config (Dict): New configuration for the rule
        
    Returns:
        bool: True if successful, False otherwise
    """
    if rule_name in VALIDATION_RULES:
        try:
            VALIDATION_RULES[rule_name] = rule_config
            return True
        except Exception as e:
            st.error(f"Error updating validation rule: {e}")
            return False
    else:
        st.error(f"Validation rule '{rule_name}' does not exist")
        return False

def remove_validation_rule(rule_name: str) -> bool:
    """Remove a validation rule.
    
    Args:
        rule_name (str): Name of the rule to remove
        
    Returns:
        bool: True if successful, False otherwise
    """
    if rule_name in VALIDATION_RULES:
        try:
            del VALIDATION_RULES[rule_name]
            return True
        except Exception as e:
            st.error(f"Error removing validation rule: {e}")
            return False
    else:
        st.error(f"Validation rule '{rule_name}' does not exist")
        return False

def run_compliance_check(content: str, compliance_framework: str = \"generic\") -> Dict:\n    \"\"\"Run a compliance check against a specific framework.\n    \n    Args:\n        content (str): Content to check\n        compliance_framework (str): Compliance framework to use\n        \n    Returns:\n        Dict: Compliance check results\n    \"\"\"\n    # Get the rule configuration\n    rule = VALIDATION_RULES.get(compliance_framework, VALIDATION_RULES.get(\"generic\", {}))\n    \n    # Apply the rule to the content\n    errors = []\n    warnings = []\n    suggestions = []\n    \n    # Check length constraints\n    if \"max_length\" in rule and len(content) > rule[\"max_length\"]:\n        errors.append(f\"Content exceeds maximum length of {rule['max_length']} characters\")\n    \n    if \"min_length\" in rule and len(content) < rule[\"min_length\"]:\n        errors.append(f\"Content is below minimum length of {rule['min_length']} characters\")\n    \n    # Check required sections\n    if \"required_sections\" in rule:\n        missing_sections = []\n        for section in rule[\"required_sections\"]:\n            if section.lower() not in content.lower():\n                missing_sections.append(section)\n        if missing_sections:\n            errors.append(f\"Missing required sections: {', '.join(missing_sections)}\")\n    \n    # Check required keywords\n    if \"required_keywords\" in rule:\n        missing_keywords = []\n        for keyword in rule[\"required_keywords\"]:\n            if keyword.lower() not in content.lower():\n                missing_keywords.append(keyword)\n        if missing_keywords:\n            warnings.append(f\"Consider adding these keywords: {', '.join(missing_keywords)}\")\n    \n    # Check forbidden patterns\n    if \"forbidden_patterns\" in rule:\n        for pattern in rule[\"forbidden_patterns\"]:\n            matches = re.findall(pattern, content)\n            if matches:\n                errors.append(f\"Forbidden pattern found: {matches[0][:50]}...\")\n    \n    return {\n        \"valid\": len(errors) == 0,\n        \"errors\": errors,\n        \"warnings\": warnings,\n        \"suggestions\": suggestions,\n        \"rule_name\": compliance_framework,\n        \"rule_config\": rule\n    }\n\ndef generate_content_summary(content: str) -> Dict:\n    \"\"\"Generate a summary of the content including key metrics.\n    \n    Args:\n        content (str): Content to summarize\n        \n    Returns:\n        Dict: Summary metrics\n    \"\"\"\n    if not content:\n        return {\n            \"word_count\": 0,\n            \"character_count\": 0,\n            \"sentence_count\": 0,\n            \"paragraph_count\": 0,\n            \"readability_score\": 0,\n            \"complexity_score\": 0\n        }\n\n    # Calculate metrics\n    complexity_metrics = calculate_protocol_complexity(content)\n    structure_analysis = extract_protocol_structure(content)\n    \n    # Simple readability calculation (Flesch Reading Ease approximation)\n    avg_sentence_length = complexity_metrics[\"avg_sentence_length\"]\n    avg_syllables_per_word = 1.3  # Rough approximation\n    readability_score = 206.835 - (1.015 * avg_sentence_length) - (84.6 * avg_syllables_per_word)\n    readability_score = max(0, min(100, readability_score))  # Clamp to 0-100 range\n    \n    return {\n        \"word_count\": complexity_metrics[\"word_count\"],\n        \"character_count\": len(content),\n        \"sentence_count\": complexity_metrics[\"sentence_count\"],\n        \"paragraph_count\": complexity_metrics[\"paragraph_count\"],\n        \"readability_score\": round(readability_score, 1),\n        \"complexity_score\": complexity_metrics[\"complexity_score\"],\n        \"has_headers\": structure_analysis[\"has_headers\"],\n        \"has_numbered_steps\": structure_analysis[\"has_numbered_steps\"],\n        \"has_bullet_points\": structure_analysis[\"has_bullet_points\"],\n        \"has_preconditions\": structure_analysis[\"has_preconditions\"],\n        \"has_postconditions\": structure_analysis[\"has_postconditions\"],\n        \"has_error_handling\": structure_analysis[\"has_error_handling\"]\n    }\n\n# Report templates\nREPORT_TEMPLATES = {\n    \"executive_summary\": {\n        \"name\": \"Executive Summary Report\",\n        \"description\": \"High-level summary of findings and recommendations\",\n        \"format\": \"markdown\",\n        \"sections\": [\"Executive Summary\", \"Key Findings\", \"Recommendations\", \"Conclusion\"]\n    },\n    \"technical_analysis\": {\n        \"name\": \"Technical Analysis Report\",\n        \"description\": \"Detailed technical analysis with implementation details\",\n        \"format\": \"markdown\",\n        \"sections\": [\"Introduction\", \"Methodology\", \"Analysis\", \"Results\", \"Implementation\", \"Conclusion\"]\n    },\n    \"security_audit\": {\n        \"name\": \"Security Audit Report\",\n        \"description\": \"Comprehensive security audit with vulnerabilities and remediation\",\n        \"format\": \"markdown\",\n        \"sections\": [\"Audit Scope\", \"Methodology\", \"Vulnerabilities\", \"Risk Assessment\", \"Remediation\", \"Conclusion\"]\n    },\n    \"compliance_review\": {\n        \"name\": \"Compliance Review Report\",\n        \"description\": \"Review of compliance with regulations and standards\",\n        \"format\": \"markdown\",\n        \"sections\": [\"Compliance Framework\", \"Assessment\", \"Findings\", \"Remediation Plan\", \"Conclusion\"]\n    },\n    \"performance_evaluation\": {\n        \"name\": \"Performance Evaluation Report\",\n        \"description\": \"Evaluation of system or process performance\",\n        \"format\": \"markdown\",\n        \"sections\": [\"Objectives\", \"Metrics\", \"Analysis\", \"Findings\", \"Improvements\", \"Conclusion\"]\n    }\n}\n\ndef list_report_templates() -> List[str]:\n    \"\"\"List all available report templates.\n    \n    Returns:\n        List[str]: List of report template names\n    \"\"\"\n    return list(REPORT_TEMPLATES.keys())\n\ndef get_report_template_details(template_name: str) -> Optional[Dict]:\n    \"\"\"Get details for a specific report template.\n    \n    Args:\n        template_name (str): Name of the template\n        \n    Returns:\n        Optional[Dict]: Template details or None if not found\n    \"\"\"\n    return REPORT_TEMPLATES.get(template_name)\n\ndef generate_custom_report(template_name: str, data: Dict) -> str:\n    \"\"\"Generate a custom report based on a template.\n    \n    Args:\n        template_name (str): Name of the template to use\n        data (Dict): Data to populate the report\n        \n    Returns:\n        str: Generated report content\n    \"\"\"\n    template = get_report_template_details(template_name)\n    if not template:\n        return f\"# Error: Template '{template_name}' not found\\n\\nUnable to generate report.\"\n\n    report_content = f\"# {template['name']}\\n\\n\"\n    report_content += f\"*Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\\n\\n\"\n    \n    # Add sections\n    for section in template.get(\"sections\", []):\n        report_content += f\"## {section}\\n\\n\"\n        \n        # Add data specific to this section\n        section_key = section.lower().replace(\" \", \"_\").replace(\"-\", \"_\")\n        if section_key in data:\n            section_data = data[section_key]\n            if isinstance(section_data, list):\n                for item in section_data:\n                    report_content += f\"- {item}\\n\"\n                report_content += \"\\n\"\n            elif isinstance(section_data, dict):\n                for key, value in section_data.items():\n                    report_content += f\"**{key}:** {value}\\n\\n\"\n            else:\n                report_content += f\"{section_data}\\n\\n\"\n        else:\n            report_content += \"*(Content to be added)*\\n\\n\"\n    \n    return report_content
