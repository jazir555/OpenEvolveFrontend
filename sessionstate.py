# ------------------------------------------------------------------
# 4. Session-state helpers
# ------------------------------------------------------------------

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
        "last_activity": _now_ms(),
        "chat_messages": [],
        "notifications": [],
        "shared_cursor_position": 0,
        "edit_locks": {}
    }

# Configuration profiles
CONFIG_PROFILES = {
    "Security Hardening": {
        "system_prompt": "You are a security expert focused on making protocols robust against attacks. Focus on identifying and closing security gaps, enforcing least privilege, and adding comprehensive error handling.",
        "evaluator_system_prompt": "You are a security auditor evaluating the protocol for vulnerabilities and weaknesses.",
        "temperature": 0.8,
        "top_p": 0.9,
        "max_iterations": 15,
        "adversarial_confidence": 95,
        "adversarial_min_iter": 5,
        "adversarial_max_iter": 20
    },
    "Compliance Focus": {
        "system_prompt": "You are a compliance expert ensuring protocols meet regulatory requirements. Focus on completeness, auditability, and regulatory alignment.",
        "evaluator_system_prompt": "You are a compliance auditor checking if the protocol meets all necessary regulatory requirements.",
        "temperature": 0.5,
        "top_p": 0.8,
        "max_iterations": 10,
        "adversarial_confidence": 90,
        "adversarial_min_iter": 3,
        "adversarial_max_iter": 15
    },
    "Operational Efficiency": {
        "system_prompt": "You are an operations expert focused on making protocols efficient and practical. Focus on streamlining processes while maintaining effectiveness.",
        "evaluator_system_prompt": "You are an operations expert evaluating the protocol for practicality and efficiency.",
        "temperature": 0.6,
        "top_p": 0.85,
        "max_iterations": 12,
        "adversarial_confidence": 85,
        "adversarial_min_iter": 3,
        "adversarial_max_iter": 12
    },
    "Beginner-Friendly": {
        "system_prompt": "You are helping a beginner write clear, understandable protocols. Focus on clarity, simplicity, and completeness.",
        "evaluator_system_prompt": "You are evaluating if the protocol is clear and understandable for beginners.",
        "temperature": 0.7,
        "top_p": 1.0,
        "max_iterations": 8,
        "adversarial_confidence": 80,
        "adversarial_min_iter": 2,
        "adversarial_max_iter": 10
    }
}

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
- [ ] Industry-specific regulations met

## Review and Approval
- Security Reviewer: [Name]
- Review Date: [Date]
- Approval Status: [Approved/Rejected/Pending]
- Notes: [Additional comments]""",
    
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

## Review and Improvement
- Retrospectives: [Schedule and process]
- KPI tracking: [Metrics monitored]
- Continuous improvement: [Process for implementing changes]""",
    
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
- Continuous improvement: [Process for enhancing the framework]""",
    
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
- Awareness Program: [How to keep personnel informed]

## Plan Distribution
[List of plan recipients and distribution methods]

## Plan Activation and Deactivation
- Activation Criteria: [When to activate the plan]
- Deactivation Criteria: [When to deactivate the plan]
- Activation Procedures: [How to activate the plan]
- Deactivation Procedures: [How to deactivate the plan]""",
    
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
- Review Schedule: [How often the process is reviewed]
- Improvement Process: [How improvements are made]""",
    
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
[How the process is reviewed and improved]"""
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
        
        session["conflict_resolutions"].append(conflict_record)
        
        return {
            "success": True,
            "conflict_detected": True,
            "conflict_record": conflict_record,
            "message": "Conflict detected. Please resolve before continuing."
        }
    else:
        # Apply operation
        session["edit_operations"].append(operation)
        
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
    
    function leaveSession(sessionId) {{
        // In a real implementation, this would leave the session
        alert('Leaving session: ' + sessionId);
    }}
    
    function saveChanges() {{
        // In a real implementation, this would save changes
        let text = editor.value;
        alert('Saving changes...');
    }}
    
    function refreshView() {{
        // In a real implementation, this would refresh the view
        alert('Refreshing view...');
    }}
    
    function acceptNew() {{
        conflictPanel.style.display = 'none';
        alert('Accepting your changes...');
    }}
    
    function acceptExisting() {{
        conflictPanel.style.display = 'none';
        alert('Accepting their changes...');
    }}
    
    function mergeChanges() {{
        conflictPanel.style.display = 'none';
        alert('Merging changes...');
    }}
    
    // Simulate real-time updates
    setInterval(function() {{
        // In a real implementation, this would fetch updates
        console.log('Checking for updates...');
    }}, 5000);
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
            <div style="background: {'linear-gradient(135deg, #4caf50, #81c784)' if results['score'] >= 80 else 'linear-gradient(135deg, #ff9800, #ffb74d)' if results['score'] >= 60 else 'linear-gradient(135deg, #f44336, #e57373)'}; 
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
    report_content += f"*Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n\n"
    
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
            <div style="position: absolute; left: -20px; top: 5px; width: 12px; height: 12px; border-radius: 50%; background-color: {'#4a6fa5' if is_current else '#6b8cbc'}; border: 2px solid white;"></div>
            <div style="background-color: {'#e3f2fd' if is_current else 'white'}; padding: 15px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); border-left: 4px solid {'#4a6fa5' if is_current else '#6b8cbc'};">
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
def list_external_integrations() -> List[str]:
    """List all available external integrations.
    
    Returns:
        List[str]: List of integration names
    """
    return list(EXTERNAL_INTEGRATIONS.keys())


def get_integration_details(integration_name: str) -> Optional[Dict]:
    """Get details for a specific external integration.
    
    Args:
        integration_name (str): Name of the integration
        
    Returns:
        Optional[Dict]: Integration details or None if not found
    """
    return EXTERNAL_INTEGRATIONS.get(integration_name)


def authenticate_integration(integration_name: str, credentials: Dict) -> bool:
    """Authenticate with an external integration.
    
    Args:
        integration_name (str): Name of the integration
        credentials (Dict): Authentication credentials
        
    Returns:
        bool: True if authentication successful, False otherwise
    """
    integration = get_integration_details(integration_name)
    if not integration:
        st.error(f"Integration '{integration_name}' not found")
        return False
    
    # In a real implementation, this would make API calls to authenticate
    # For now, we'll just simulate success
    st.session_state[f"{integration_name}_authenticated"] = True
    st.session_state[f"{integration_name}_credentials"] = credentials
    
    st.success(f"Successfully authenticated with {integration['name']}")
    return True


def publish_to_github(repository: str, file_path: str, content: str, commit_message: str, token: str) -> Dict:
    """Publish content to a GitHub repository.
    
    Args:
        repository (str): Repository name (owner/repo)
        file_path (str): Path to the file in the repository
        content (str): Content to publish
        commit_message (str): Commit message
        token (str): GitHub personal access token
        
    Returns:
        Dict: Publication result
    """
    try:
        # In a real implementation, this would make API calls to GitHub
        # For now, we'll simulate the process
        result = {
            "success": True,
            "repository": repository,
            "file_path": file_path,
            "commit_sha": "abc123def456",  # Simulated commit SHA
            "url": f"https://github.com/{repository}/blob/main/{file_path}",
            "timestamp": datetime.now().isoformat()
        }
        
        st.success(f"Successfully published to GitHub: {repository}/{file_path}")
        return result
    except Exception as e:
        st.error(f"Error publishing to GitHub: {e}")
        return {"success": False, "error": str(e)}


def create_gitlab_issue(project_id: str, title: str, description: str, token: str) -> Dict:
    """Create an issue in a GitLab project.
    
    Args:
        project_id (str): GitLab project ID
        title (str): Issue title
        description (str): Issue description
        token (str): GitLab personal access token
        
    Returns:
        Dict: Issue creation result
    """
    try:
        # In a real implementation, this would make API calls to GitLab
        # For now, we'll simulate the process
        result = {
            "success": True,
            "project_id": project_id,
            "issue_id": "12345",  # Simulated issue ID
            "title": title,
            "url": f"https://gitlab.com/{project_id}/-/issues/12345",
            "timestamp": datetime.now().isoformat()
        }
        
        st.success(f"Successfully created GitLab issue: {title}")
        return result
    except Exception as e:
        st.error(f"Error creating GitLab issue: {e}")
        return {"success": False, "error": str(e)}


def post_to_slack(channel: str, message: str, token: str) -> Dict:
    """Post a message to a Slack channel.
    
    Args:
        channel (str): Slack channel name or ID
        message (str): Message to post
        token (str): Slack bot token
        
    Returns:
        Dict: Posting result
    """
    try:
        # In a real implementation, this would make API calls to Slack
        # For now, we'll simulate the process
        result = {
            "success": True,
            "channel": channel,
            "message": message,
            "timestamp": datetime.now().isoformat(),
            "message_id": "xyz789"  # Simulated message ID
        }
        
        st.success(f"Successfully posted to Slack channel: {channel}")
        return result
    except Exception as e:
        st.error(f"Error posting to Slack: {e}")
        return {"success": False, "error": str(e)}


def render_integration_manager_ui() -> str:
    """Render the external integration manager UI.
    
    Returns:
        str: HTML formatted integration manager UI
    """
    html = """
    <div style="background-color: #f8f9fa; padding: 20px; border-radius: 10px; margin-bottom: 20px;">
        <h2 style="color: #4a6fa5; margin-top: 0; text-align: center;">🔌 External Integrations</h2>
        
        <!-- Integration Cards -->
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin-bottom: 20px;">
    """
    
    # Add integration cards
    for integration_name, integration in EXTERNAL_INTEGRATIONS.items():
        is_authenticated = st.session_state.get(f"{integration_name}_authenticated", False)
        
        html += f"""
        <div style="background-color: white; border-radius: 10px; padding: 15px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); border: 1px solid #eee;">
            <h3 style="color: #4a6fa5; margin-top: 0;">
                {integration['name']}
                <span style="float: right; font-size: 1.2em;">{'✅' if is_authenticated else '❌'}</span>
            </h3>
            <p style="color: #666; font-size: 0.9em; margin-bottom: 15px;">{integration['description']}</p>
            <div style="display: flex; flex-wrap: wrap; gap: 5px; margin-bottom: 15px;">
    """
        
        # Add capability tags
        for capability in integration.get("capabilities", []):
            html += f"""
                <span style="background-color: #e3f2fd; color: #1565c0; padding: 3px 8px; border-radius: 10px; font-size: 0.7em;">
                    {capability.replace('_', ' ').title()}
                </span>
            """
        
        html += """
            </div>
            <button onclick="configureIntegration('{integration_name}')" 
                    style="width: 100%; background-color: #4a6fa5; color: white; border: none; padding: 8px; border-radius: 5px; cursor: pointer;">
                {auth_button_text}
            </button>
        </div>
        """.format(
            integration_name=integration_name,
            auth_button_text="Configure" if not is_authenticated else "Reconfigure"
        )
    
    html += """
        </div>
        
        <!-- Configuration Panel -->
        <div id="integrationConfig" style="background-color: white; padding: 15px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); display: none;">
            <h3 style="color: #4a6fa5; margin-top: 0;">Configure Integration</h3>
            <div id="configForm">
                <!-- Configuration form will be populated by JavaScript -->
            </div>
        </div>
        
        <!-- Action Panel -->
        <div id="integrationActions" style="background-color: white; padding: 15px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin-top: 20px; display: none;">
            <h3 style="color: #4a6fa5; margin-top: 0;">Integration Actions</h3>
            <div id="actionButtons">
                <!-- Action buttons will be populated by JavaScript -->
            </div>
        </div>
    </div>
    
    <script>
    let selectedIntegration = null;
    
    function configureIntegration(integrationName) {
        selectedIntegration = integrationName;
        document.getElementById('integrationConfig').style.display = 'block';
        
        // Populate configuration form
        const formContainer = document.getElementById('configForm');
        
        // In a real implementation, this would dynamically generate forms
        // based on the integration requirements
        formContainer.innerHTML = `
            <div style="margin-bottom: 15px;">
                <label style="display: block; margin-bottom: 5px; font-weight: bold;">Authentication Token</label>
                <input type="password" id="authToken" style="width: 100%; padding: 8px; border: 1px solid #ddd; border-radius: 5px;" placeholder="Enter your authentication token">
                <p style="font-size: 0.8em; color: #666; margin-top: 5px;">
                    Generate a token in your ${integrationName.charAt(0).toUpperCase() + integrationName.slice(1)} account settings
                </p>
            </div>
            <div style="margin-bottom: 15px;">
                <button onclick="saveConfiguration()" style="background-color: #4a6fa5; color: white; border: none; padding: 10px 20px; border-radius: 5px; cursor: pointer;">
                    Save Configuration
                </button>
                <button onclick="testConnection()" style="background-color: #6b8cbc; color: white; border: none; padding: 10px 20px; border-radius: 5px; cursor: pointer; margin-left: 10px;">
                    Test Connection
                </button>
            </div>
        `;
        
        // Show action panel
        document.getElementById('integrationActions').style.display = 'block';
        updateActionButtons(integrationName);
    }
    
    function updateActionButtons(integrationName) {
        const actionContainer = document.getElementById('actionButtons');
        
        // Different actions based on integration
        let actionsHtml = '';
        switch(integrationName) {
            case 'github':
                actionsHtml = `
                    <button onclick="publishToRepo()" style="background-color: #4a6fa5; color: white; border: none; padding: 10px 15px; border-radius: 5px; cursor: pointer; margin-right: 10px;">
                        Publish to Repository
                    </button>
                    <button onclick="createIssue()" style="background-color: #6b8cbc; color: white; border: none; padding: 10px 15px; border-radius: 5px; cursor: pointer;">
                        Create Issue
                    </button>
                `;
                break;
            case 'gitlab':
                actionsHtml = `
                    <button onclick="createGitlabIssue()" style="background-color: #4a6fa5; color: white; border: none; padding: 10px 15px; border-radius: 5px; cursor: pointer;">
                        Create Issue
                    </button>
                `;
                break;
            case 'slack':
                actionsHtml = `
                    <button onclick="postToChannel()" style="background-color: #4a6fa5; color: white; border: none; padding: 10px 15px; border-radius: 5px; cursor: pointer;">
                        Post to Channel
                    </button>
                `;
                break;
            default:
                actionsHtml = `
                    <button onclick="genericAction()" style="background-color: #4a6fa5; color: white; border: none; padding: 10px 15px; border-radius: 5px; cursor: pointer;">
                        Perform Action
                    </button>
                `;
        }
        
        actionContainer.innerHTML = actionsHtml;
    }
    
    function saveConfiguration() {
        const token = document.getElementById('authToken').value;
        if (!token) {
            alert('Please enter an authentication token.');
            return;
        }
        
        // In a real implementation, this would save the configuration
        alert(`Configuration saved for ${selectedIntegration}`);
    }
    
    function testConnection() {
        const token = document.getElementById('authToken').value;
        if (!token) {
            alert('Please enter an authentication token.');
            return;
        }
        
        // In a real implementation, this would test the connection
        alert(`Connection test successful for ${selectedIntegration}`);
    }
    
    function publishToRepo() {
        // In a real implementation, this would show a publish dialog
        alert('Publish to repository dialog would appear here');
    }
    
    function createIssue() {
        // In a real implementation, this would show an issue creation dialog
        alert('Create issue dialog would appear here');
    }
    
    function createGitlabIssue() {
        // In a real implementation, this would show a GitLab issue creation dialog
        alert('Create GitLab issue dialog would appear here');
    }
    
    function postToChannel() {
        // In a real implementation, this would show a Slack post dialog
        alert('Post to Slack channel dialog would appear here');
    }
    
    function genericAction() {
        alert(`Performing generic action for ${selectedIntegration}`);
    }
    </script>
    """
    
    return html


# Machine Learning-Based Protocol Suggestions
def list_ml_models() -> List[str]:
    """List all available ML models for protocol suggestions.
    
    Returns:
        List[str]: List of model names
    """
    return list(ML_SUGGESTION_MODELS.keys())


def get_model_details(model_name: str) -> Optional[Dict]:
    """Get details for a specific ML model.
    
    Args:
        model_name (str): Name of the model
        
    Returns:
        Optional[Dict]: Model details or None if not found
    """
    return ML_SUGGESTION_MODELS.get(model_name)


def generate_ml_suggestions(model_name: str, protocol_text: str) -> Dict:
    """Generate protocol suggestions using an ML model.
    
    Args:
        model_name (str): Name of the ML model to use
        protocol_text (str): Protocol text to analyze
        
    Returns:
        Dict: Suggestions and improvements
    """
    model = get_model_details(model_name)
    if not model:
        return {"success": False, "error": f"Model '{model_name}' not found"}
    
    # In a real implementation, this would call an ML API
    # For now, we'll simulate suggestions based on the model type
    suggestions = []
    
    if model_name == "protocol_improver":
        suggestions = [
            "Consider adding a section on roles and responsibilities for better clarity",
            "Add more specific examples to illustrate key concepts",
            "Include a glossary of terms for complex terminology",
            "Add cross-references to related sections for better navigation",
            "Consider adding diagrams or flowcharts to visualize complex processes"
        ]
    elif model_name == "compliance_checker":
        suggestions = [
            {
                "issue": "Missing GDPR compliance section",
                "severity": "high",
                "recommendation": "Add a section detailing GDPR compliance requirements"
            },
            {
                "issue": "Lack of data retention policy",
                "severity": "medium",
                "recommendation": "Specify how long data will be retained and deletion procedures"
            },
            {
                "issue": "No individual rights section",
                "severity": "high",
                "recommendation": "Include a section on individual rights under applicable regulations"
            }
        ]
    elif model_name == "security_analyzer":
        suggestions = [
            {
                "issue": "Weak authentication requirements",
                "severity": "high",
                "category": "access_control",
                "recommendation": "Specify multi-factor authentication requirements for sensitive systems"
            },
            {
                "issue": "Lack of encryption specifications",
                "severity": "medium",
                "category": "data_protection",
                "recommendation": "Define encryption standards for data at rest and in transit"
            },
            {
                "issue": "Insufficient incident response procedures",
                "severity": "high",
                "category": "incident_response",
                "recommendation": "Add detailed incident response procedures with escalation paths"
            }
        ]
    elif model_name == "readability_enhancer":
        suggestions = [
            "Shorten sentences longer than 25 words for better readability",
            "Replace complex jargon with simpler terms where possible",
            "Add subheadings to break up long sections",
            "Use bullet points or numbered lists for procedural steps",
            "Ensure consistent terminology throughout the document"
        ]
    
    return {
        "success": True,
        "model": model_name,
        "model_details": model,
        "suggestions": suggestions,
        "timestamp": datetime.now().isoformat()
    }


def apply_ml_suggestions(protocol_text: str, suggestions: List[Dict]) -> str:
    """Apply ML-generated suggestions to improve a protocol.
    
    Args:
        protocol_text (str): Original protocol text
        suggestions (List[Dict]): List of suggestions to apply
        
    Returns:
        str: Improved protocol text
    """
    # In a real implementation, this would intelligently apply suggestions
    # For now, we'll simulate by adding a note about the suggestions
    improved_text = protocol_text + "\n\n"
    improved_text += "# AI-Generated Improvements\n\n"
    improved_text += "The following improvements were suggested by AI analysis:\n\n"
    
    for i, suggestion in enumerate(suggestions, 1):
        if isinstance(suggestion, str):
            improved_text += f"{i}. {suggestion}\n"
        elif isinstance(suggestion, dict):
            improved_text += f"{i}. {suggestion.get('recommendation', suggestion.get('issue', 'Improvement'))}\n"
    
    return improved_text


def render_ml_suggestions_ui(protocol_text: str) -> str:
    """Render the ML suggestions UI.
    
    Args:
        protocol_text (str): Protocol text to analyze
        
    Returns:
        str: HTML formatted suggestions UI
    """
    html = """
    <div style="background-color: #f8f9fa; padding: 20px; border-radius: 10px; margin-bottom: 20px;">
        <h2 style="color: #4a6fa5; margin-top: 0; text-align: center;">🤖 AI-Powered Protocol Suggestions</h2>
        
        <!-- Model Selection -->
        <div style="background-color: white; padding: 15px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin-bottom: 20px;">
            <h3 style="color: #4a6fa5; margin-top: 0;">Select AI Analysis Model</h3>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px;">
    """
    
    # Add model cards
    for model_name, model in ML_SUGGESTION_MODELS.items():
        html += f"""
        <div style="background-color: #e8f5e9; border-radius: 8px; padding: 15px; cursor: pointer; border: 2px solid #c8e6c9;" 
             onclick="analyzeWithModel('{model_name}')">
            <h4 style="color: #2e7d32; margin-top: 0;">{model['name']}</h4>
            <p style="color: #388e3c; font-size: 0.9em; margin-bottom: 10px;">{model['description']}</p>
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <span style="background-color: #a5d6a7; color: #1b5e20; padding: 3px 8px; border-radius: 10px; font-size: 0.8em;">
                    {model['input_format'].upper()}
                </span>
                <span style="font-size: 0.8em; color: #388e3c;">
                    → {model['output_format'].upper()}
                </span>
            </div>
        </div>
        """
    
    html += """
            </div>
        </div>
        
        <!-- Analysis Results -->
        <div id="analysisResults" style="background-color: white; padding: 15px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); display: none;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px;">
                <h3 style="color: #4a6fa5; margin: 0;">Analysis Results</h3>
                <button onclick="applySuggestions()" style="background-color: #4caf50; color: white; border: none; padding: 8px 15px; border-radius: 5px; cursor: pointer;">
                    Apply Suggestions
                </button>
            </div>
            <div id="suggestionsList" style="max-height: 400px; overflow-y: auto;">
                <!-- Suggestions will be populated by JavaScript -->
            </div>
        </div>
        
        <!-- Applied Improvements -->
        <div id="appliedImprovements" style="background-color: #e3f2fd; padding: 15px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin-top: 20px; display: none;">
            <h3 style="color: #1565c0; margin-top: 0;">✨ Improvements Applied</h3>
            <p id="improvementSummary">AI suggestions have been applied to your protocol.</p>
            <div style="margin-top: 10px;">
                <button onclick="viewImprovedProtocol()" style="background-color: #2196f3; color: white; border: none; padding: 8px 15px; border-radius: 5px; cursor: pointer; margin-right: 10px;">
                    View Improved Protocol
                </button>
                <button onclick="undoImprovements()" style="background-color: #ff9800; color: white; border: none; padding: 8px 15px; border-radius: 5px; cursor: pointer;">
                    Undo Changes
                </button>
            </div>
        </div>
    </div>
    
    <script>
    let currentSuggestions = [];
    
    function analyzeWithModel(modelName) {
        // In a real implementation, this would call the ML API
        // For now, we'll simulate results
        const resultsContainer = document.getElementById('analysisResults');
        const suggestionsContainer = document.getElementById('suggestionsList');
        
        // Show loading state
        suggestionsContainer.innerHTML = '<p style="text-align: center; padding: 20px;">🔬 Analyzing protocol with AI model...</p>';
        resultsContainer.style.display = 'block';
        
        // Simulate API delay
        setTimeout(() => {
            // Generate simulated suggestions based on model
            let suggestions = [];
            switch(modelName) {
                case 'protocol_improver':
                    suggestions = [
                        "Consider adding a section on roles and responsibilities for better clarity",
                        "Add more specific examples to illustrate key concepts",
                        "Include a glossary of terms for complex terminology",
                        "Add cross-references to related sections for better navigation",
                        "Consider adding diagrams or flowcharts to visualize complex processes"
                    ];
                    break;
                case 'compliance_checker':
                    suggestions = [
                        {
                            "issue": "Missing GDPR compliance section",
                            "severity": "high",
                            "recommendation": "Add a section detailing GDPR compliance requirements"
                        },
                        {
                            "issue": "Lack of data retention policy",
                            "severity": "medium",
                            "recommendation": "Specify how long data will be retained and deletion procedures"
                        },
                        {
                            "issue": "No individual rights section",
                            "severity": "high",
                            "recommendation": "Include a section on individual rights under applicable regulations"
                        }
                    ];
                    break;
                case 'security_analyzer':
                    suggestions = [
                        {
                            "issue": "Weak authentication requirements",
                            "severity": "high",
                            "category": "access_control",
                            "recommendation": "Specify multi-factor authentication requirements for sensitive systems"
                        },
                        {
                            "issue": "Lack of encryption specifications",
                            "severity": "medium",
                            "category": "data_protection",
                            "recommendation": "Define encryption standards for data at rest and in transit"
                        },
                        {
                            "issue": "Insufficient incident response procedures",
                            "severity": "high",
                            "category": "incident_response",
                            "recommendation": "Add detailed incident response procedures with escalation paths"
                        }
                    ];
                    break;
                case 'readability_enhancer':
                    suggestions = [
                        "Shorten sentences longer than 25 words for better readability",
                        "Replace complex jargon with simpler terms where possible",
                        "Add subheadings to break up long sections",
                        "Use bullet points or numbered lists for procedural steps",
                        "Ensure consistent terminology throughout the document"
                    ];
                    break;
            }
            
            currentSuggestions = suggestions;
            
            // Render suggestions
            let suggestionsHtml = '<div style="display: grid; gap: 15px;">';
            
            suggestions.forEach((suggestion, index) => {
                if (typeof suggestion === 'string') {
                    suggestionsHtml += `
                        <div style="background-color: #fff8e1; padding: 15px; border-radius: 8px; border-left: 4px solid #ff9800;">
                            <div style="display: flex; align-items: flex-start;">
                                <span style="margin-right: 10px; font-size: 1.2em;">💡</span>
                                <div>
                                    <p style="margin: 0;">${suggestion}</p>
                                </div>
                            </div>
                        </div>
                    `;
                } else {
                    const severityColor = {
                        'low': '#4caf50',
                        'medium': '#ff9800',
                        'high': '#f44336',
                        'critical': '#9c27b0'
                    }[suggestion.severity] || '#666';
                    
                    suggestionsHtml += `
                        <div style="background-color: #fff8e1; padding: 15px; border-radius: 8px; border-left: 4px solid ${severityColor};">
                            <div style="display: flex; align-items: flex-start;">
                                <span style="margin-right: 10px; font-size: 1.2em;">${suggestion.severity === 'high' || suggestion.severity === 'critical' ? '⚠️' : 'ℹ️'}</span>
                                <div style="flex: 1;">
                                    <p style="margin: 0; font-weight: bold; color: ${severityColor};">${suggestion.issue}</p>
                                    <p style="margin: 5px 0 0 0;">${suggestion.recommendation}</p>
                                    ${suggestion.category ? `<span style="background-color: #bbdefb; color: #0d47a1; padding: 2px 6px; border-radius: 10px; font-size: 0.8em; margin-top: 5px; display: inline-block;">${suggestion.category.replace('_', ' ')}</span>` : ''}
                                </div>
                            </div>
                        </div>
                    `;
                }
            });
            
            suggestionsHtml += '</div>';
            suggestionsContainer.innerHTML = suggestionsHtml;
        }, 1500);
    }
    
    function applySuggestions() {
        if (currentSuggestions.length === 0) {
            alert('No suggestions to apply.');
            return;
        }
        
        // In a real implementation, this would apply the suggestions to the protocol
        document.getElementById('appliedImprovements').style.display = 'block';
        document.getElementById('improvementSummary').textContent = 
            `Applied ${currentSuggestions.length} AI suggestions to improve your protocol.`;
        
        // Hide the analysis results
        document.getElementById('analysisResults').style.display = 'none';
        
        alert('AI suggestions applied successfully!');
    }
    
    function viewImprovedProtocol() {
        alert('Viewing improved protocol... (This would show the updated protocol)');
    }
    
    function undoImprovements() {
        document.getElementById('appliedImprovements').style.display = 'none';
        alert('AI improvements undone.');
    }
    </script>
    """
    
    return html


# External Integration Configuration
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
    
    complexity_score = max(0, 100 - complexity["complexity_score"])  # Invert complexity
    
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