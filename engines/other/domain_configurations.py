"""
Domain Configurations for Decomposition System

Provides predefined configurations for various domains with domain-specific
terminology, patterns, strategies, and quality thresholds.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any


@dataclass
class DomainConfiguration:
    """Configuration for a specific domain."""
    domain: str
    domain_name: str

    # Strategies
    preferred_strategies: List[str] = field(default_factory=list)
    avoided_strategies: List[str] = field(default_factory=list)
    strategy_weights: Dict[str, float] = field(default_factory=dict)  # Overrides default weights

    # Patterns
    common_patterns: List[str] = field(default_factory=list)
    decomposition_approaches: List[str] = field(default_factory=list)

    # Vocabulary
    terminology: Dict[str, str] = field(default_factory=dict)  # term -> definition
    common_phrases: List[str] = field(default_factory=list)

    # Complexity
    typical_complexity: float = 0.5  # 0-1
    complexity_distribution: Dict[str, float] = field(default_factory=dict)  # low/medium/high percentages

    # Quality
    quality_thresholds: Dict[str, float] = field(default_factory=dict)
    quality_dimensions_importance: Dict[str, float] = field(default_factory=dict)

    # Teams
    required_expertise: List[str] = field(default_factory=list)
    team_preferences: Dict[str, List[str]] = field(default_factory=dict)  # role -> preferred team IDs

    # Resources
    resource_multipliers: Dict[str, float] = field(default_factory=dict)
    typical_effort_multipliers: Dict[str, float] = field(default_factory=dict)

    def validate(self) -> List[str]:
        """Validate domain configuration."""
        from typing import List
        errors = []

        if not self.domain:
            errors.append("DomainConfiguration domain cannot be empty")

        if not (0.0 <= self.typical_complexity <= 1.0):
            errors.append(f"typical_complexity must be between 0.0 and 1.0, got {self.typical_complexity}")

        # Validate complexity distribution sums to 1.0
        if self.complexity_distribution:
            total = sum(self.complexity_distribution.values())
            if abs(total - 1.0) > 0.01:
                errors.append(f"complexity_distribution must sum to 1.0, got {total}")

        # Validate quality thresholds are between 0 and 1
        for name, threshold in self.quality_thresholds.items():
            if not (0.0 <= threshold <= 1.0):
                errors.append(f"quality_threshold {name} must be between 0.0 and 1.0, got {threshold}")

        # Validate quality dimensions importance sums to 1.0
        if self.quality_dimensions_importance:
            total = sum(self.quality_dimensions_importance.values())
            if abs(total - 1.0) > 0.01:
                errors.append(f"quality_dimensions_importance must sum to 1.0, got {total}")

        return errors


# Predefined domain configurations
DOMAIN_CONFIGURATIONS = {
    "machine_learning": DomainConfiguration(
        domain="machine_learning",
        domain_name="Machine Learning",
        preferred_strategies=["complexity", "functional", "technical_dependency"],
        avoided_strategies=["temporal"],
        strategy_weights={
            "complexity": 0.35,
            "functional": 0.30,
            "technical_dependency": 0.25,
            "semantic": 0.10
        },
        common_patterns=[
            "Data preprocessing is first step",
            "Model training is computational intensive",
            "Evaluation requires test dataset",
            "Deployment needs infrastructure"
        ],
        decomposition_approaches=[
            "data_first",  # Start with data preparation
            "model_centric",  # Focus on model architecture
            "pipeline_based",  # End-to-end pipeline
            "iterative_refinement"  # Iterative improvement
        ],
        terminology={
            "training": "Teaching model with data",
            "inference": "Using model for predictions",
            "overfitting": "Model memorizes training data",
            "hyperparameter": "Configuration setting for model",
            "feature_engineering": "Creating input features",
            "cross_validation": "Model validation technique",
            "gradient_descent": "Optimization algorithm",
            "epoch": "One full training pass"
        },
        common_phrases=[
            "train the model",
            "evaluate performance",
            "feature extraction",
            "model selection",
            "hyperparameter tuning"
        ],
        typical_complexity=0.7,
        complexity_distribution={
            "low": 0.2,
            "medium": 0.5,
            "high": 0.3
        },
        quality_thresholds={
            "completeness": 0.75,
            "feasibility": 0.70,
            "consistency": 0.80
        },
        quality_dimensions_importance={
            "completeness": 0.30,
            "consistency": 0.25,
            "feasibility": 0.30,
            "balance": 0.15
        },
        required_expertise=[
            "data_science",
            "statistics",
            "programming",
            "mathematics"
        ],
        team_preferences={
            "solver": ["ml_research_team", "data_science_team"],
            "patcher": ["ml_engineering_team"],
            "red_team": ["ml_validation_team"],
            "gold_team": ["ml_expert_team"]
        },
        resource_multipliers={
            "time_hours": 1.5,
            "api_tokens": 2.0,
            "computational_units": 3.0
        },
        typical_effort_multipliers={
            "research": 1.8,
            "implementation": 1.5,
            "validation": 1.3
        }
    ),

    "software_development": DomainConfiguration(
        domain="software_development",
        domain_name="Software Development",
        preferred_strategies=["functional", "technical_dependency", "complexity"],
        avoided_strategies=[],
        strategy_weights={
            "functional": 0.35,
            "technical_dependency": 0.30,
            "complexity": 0.25,
            "semantic": 0.10
        },
        common_patterns=[
            "Requirements before design",
            "Design before implementation",
            "Testing before deployment",
            "Documentation throughout"
        ],
        decomposition_approaches=[
            "feature_based",  # Decompose by features
            "layered_architecture",  # By architectural layers
            "component_based",  # By components
            "use_case_driven"  # By use cases
        ],
        terminology={
            "refactor": "Restructure code without changing behavior",
            "technical_debt": "Cost of shortcuts taken",
            "code_review": "Peer examination of code",
            "sprint": "Focused development period",
            "commit": "Save changes to version control",
            "pull_request": "Propose changes for review",
            "unit_test": "Test individual components",
            "integration_test": "Test component interactions"
        },
        common_phrases=[
            "write unit tests",
            "code review",
            "merge to main",
            "deploy to production",
            "fix bug"
        ],
        typical_complexity=0.6,
        complexity_distribution={
            "low": 0.3,
            "medium": 0.5,
            "high": 0.2
        },
        quality_thresholds={
            "completeness": 0.80,
            "feasibility": 0.75,
            "consistency": 0.85
        },
        quality_dimensions_importance={
            "completeness": 0.25,
            "consistency": 0.35,
            "feasibility": 0.25,
            "balance": 0.15
        },
        required_expertise=[
            "programming",
            "software_architecture",
            "testing",
            "devops"
        ],
        team_preferences={
            "solver": ["development_team", "engineering_team"],
            "patcher": ["maintenance_team"],
            "red_team": ["qa_team", "security_team"],
            "gold_team": ["senior_developers", "architects"]
        },
        resource_multipliers={
            "time_hours": 1.2,
            "api_tokens": 1.0,
            "computational_units": 1.0
        },
        typical_effort_multipliers={
            "research": 1.0,
            "implementation": 1.3,
            "validation": 1.1
        }
    ),

    "research": DomainConfiguration(
        domain="research",
        domain_name="Academic Research",
        preferred_strategies=["research", "complexity", "semantic"],
        avoided_strategies=[],
        strategy_weights={
            "research": 0.35,
            "complexity": 0.25,
            "semantic": 0.25,
            "functional": 0.15
        },
        common_patterns=[
            "Literature review comes first",
            "Hypothesis formulation",
            "Methodology design",
            "Data collection",
            "Analysis and synthesis",
            "Peer review and publication"
        ],
        decomposition_approaches=[
            "literature_first",  # Start with literature review
            "hypothesis_driven",  # Focus on hypothesis
            "methodology_centric",  # Focus on methods
            "iterative_inquiry"  # Iterative exploration
        ],
        terminology={
            "literature_review": "Survey of existing research",
            "hypothesis": "Testable research question",
            "methodology": "Research approach",
            "peer_review": "Evaluation by other researchers",
            "abstract": "Summary of research",
            "citation": "Reference to prior work",
            "experimental_design": "Plan for experiments",
            "statistical_significance": "Results not due to chance"
        },
        common_phrases=[
            "conduct literature review",
            "formulate hypothesis",
            "design experiment",
            "collect data",
            "analyze results"
        ],
        typical_complexity=0.8,
        complexity_distribution={
            "low": 0.1,
            "medium": 0.4,
            "high": 0.5
        },
        quality_thresholds={
            "completeness": 0.70,
            "feasibility": 0.65,
            "consistency": 0.75
        },
        quality_dimensions_importance={
            "completeness": 0.35,
            "consistency": 0.25,
            "feasibility": 0.20,
            "balance": 0.20
        },
        required_expertise=[
            "domain_knowledge",
            "research_methods",
            "statistics",
            "academic_writing"
        ],
        team_preferences={
            "solver": ["research_team", "domain_experts"],
            "patcher": ["research_assistants"],
            "red_team": ["peer_reviewers", "methodology_experts"],
            "gold_team": ["senior_researchers", "advisors"]
        },
        resource_multipliers={
            "time_hours": 2.0,
            "api_tokens": 1.5,
            "computational_units": 1.0
        },
        typical_effort_multipliers={
            "research": 2.5,
            "implementation": 1.0,
            "validation": 1.8
        }
    ),

    "devops": DomainConfiguration(
        domain="devops",
        domain_name="DevOps / Infrastructure",
        preferred_strategies=["technical_dependency", "functional", "temporal"],
        avoided_strategies=[],
        strategy_weights={
            "technical_dependency": 0.40,
            "functional": 0.25,
            "temporal": 0.25,
            "complexity": 0.10
        },
        common_patterns=[
            "Infrastructure setup",
            "CI/CD pipeline",
            "Monitoring and logging",
            "Backup and recovery",
            "Security hardening"
        ],
        decomposition_approaches=[
            "infrastructure_first",  # Start with infrastructure
            "pipeline_centric",  # Focus on CI/CD
            "service_based",  # By services
            "environment_separated"  # Dev/staging/prod
        ],
        terminology={
            "deployment": "Release to production",
            "pipeline": "Automated delivery process",
            "monitoring": "Tracking system health",
            "scaling": "Handling increased load",
            "containerization": "Packaging applications",
            "orchestration": "Managing containers",
            "infrastructure_as_code": "Code-defined infrastructure",
            "continuous_integration": "Automated testing and integration"
        },
        common_phrases=[
            "deploy to production",
            "set up monitoring",
            "configure pipeline",
            "scale infrastructure",
            "rollback deployment"
        ],
        typical_complexity=0.65,
        complexity_distribution={
            "low": 0.25,
            "medium": 0.55,
            "high": 0.20
        },
        quality_thresholds={
            "completeness": 0.85,
            "feasibility": 0.80,
            "consistency": 0.90
        },
        quality_dimensions_importance={
            "completeness": 0.25,
            "consistency": 0.40,
            "feasibility": 0.25,
            "balance": 0.10
        },
        required_expertise=[
            "system_administration",
            "cloud_platforms",
            "automation",
            "security"
        ],
        team_preferences={
            "solver": ["devops_team", "sre_team"],
            "patcher": ["infrastructure_team"],
            "red_team": ["security_team", "reliability_team"],
            "gold_team": ["sre_leads", "architects"]
        },
        resource_multipliers={
            "time_hours": 1.3,
            "api_tokens": 1.2,
            "computational_units": 2.5
        },
        typical_effort_multipliers={
            "research": 0.8,
            "implementation": 1.4,
            "validation": 1.5
        }
    ),

    "data_engineering": DomainConfiguration(
        domain="data_engineering",
        domain_name="Data Engineering",
        preferred_strategies=["technical_dependency", "functional", "complexity"],
        avoided_strategies=["temporal"],
        strategy_weights={
            "technical_dependency": 0.35,
            "functional": 0.30,
            "complexity": 0.25,
            "semantic": 0.10
        },
        common_patterns=[
            "Data ingestion first",
            "Storage before processing",
            "Transformation pipeline",
            "Quality validation",
            "Monitoring and alerting"
        ],
        decomposition_approaches=[
            "data_flow",  # Follow data flow
            "pipeline_based",  # By pipelines
            "source_centric",  # By data sources
            "layered_architecture"  # Ingestion/processing/serving
        ],
        terminology={
            "etl": "Extract, Transform, Load",
            "data_lake": "Raw data storage",
            "data_warehouse": "Structured data storage",
            "pipeline": "Data processing workflow",
            "schema": "Data structure definition",
            "partitioning": "Dividing data for performance",
            "orchestration": "Managing data workflows",
            "data_quality": "Data accuracy and completeness"
        },
        common_phrases=[
            "build pipeline",
            "transform data",
            "validate quality",
            "monitor data flow",
            "optimize query"
        ],
        typical_complexity=0.7,
        complexity_distribution={
            "low": 0.2,
            "medium": 0.5,
            "high": 0.3
        },
        quality_thresholds={
            "completeness": 0.80,
            "feasibility": 0.75,
            "consistency": 0.85
        },
        quality_dimensions_importance={
            "completeness": 0.30,
            "consistency": 0.35,
            "feasibility": 0.25,
            "balance": 0.10
        },
        required_expertise=[
            "database_design",
            "etl_tools",
            "distributed_systems",
            "data_modeling"
        ],
        team_preferences={
            "solver": ["data_engineering_team"],
            "patcher": ["pipeline_team"],
            "red_team": ["data_quality_team"],
            "gold_team": ["data_architects"]
        },
        resource_multipliers={
            "time_hours": 1.4,
            "api_tokens": 1.3,
            "computational_units": 2.0
        },
        typical_effort_multipliers={
            "research": 1.0,
            "implementation": 1.6,
            "validation": 1.4
        }
    ),

    "cybersecurity": DomainConfiguration(
        domain="cybersecurity",
        domain_name="Cybersecurity",
        preferred_strategies=["complexity", "functional", "technical_dependency"],
        avoided_strategies=[],
        strategy_weights={
            "complexity": 0.35,
            "functional": 0.30,
            "technical_dependency": 0.25,
            "semantic": 0.10
        },
        common_patterns=[
            "Threat modeling first",
            "Vulnerability assessment",
            "Security controls implementation",
            "Penetration testing",
            "Monitoring and response"
        ],
        decomposition_approaches=[
            "threat_modeling",  # Start with threats
            "defense_in_depth",  # Layered security
            "asset_based",  # By assets to protect
            "attack_vector"  # By attack vectors
        ],
        terminology={
            "vulnerability": "Security weakness",
            "exploit": "Attack using vulnerability",
            "penetration_test": "Authorized security testing",
            "threat_modeling": "Identifying potential threats",
            "encryption": "Data encoding for protection",
            "authentication": "Verifying identity",
            "authorization": "Verifying permissions",
            "incident_response": "Reacting to security breaches"
        },
        common_phrases=[
            "conduct vulnerability scan",
            "implement security controls",
            "perform penetration test",
            "monitor threats",
            "respond to incident"
        ],
        typical_complexity=0.85,
        complexity_distribution={
            "low": 0.1,
            "medium": 0.4,
            "high": 0.5
        },
        quality_thresholds={
            "completeness": 0.90,
            "feasibility": 0.75,
            "consistency": 0.85
        },
        quality_dimensions_importance={
            "completeness": 0.35,
            "consistency": 0.30,
            "feasibility": 0.20,
            "balance": 0.15
        },
        required_expertise=[
            "security_principles",
            "networking",
            "cryptography",
            "compliance"
        ],
        team_preferences={
            "solver": ["security_team", "blue_team"],
            "patcher": ["security_engineers"],
            "red_team": ["red_team", "pen_testers"],
            "gold_team": ["security_architects", "ciso_office"]
        },
        resource_multipliers={
            "time_hours": 1.6,
            "api_tokens": 1.4,
            "computational_units": 1.5
        },
        typical_effort_multipliers={
            "research": 1.5,
            "implementation": 1.4,
            "validation": 2.0
        }
    )
}


def get_domain_config(domain: str) -> DomainConfiguration:
    """
    Get domain configuration by domain name.

    Args:
        domain: Domain identifier

    Returns:
        DomainConfiguration for the domain

    Raises:
        KeyError: If domain not found
    """
    if domain not in DOMAIN_CONFIGURATIONS:
        raise KeyError(f"Domain '{domain}' not found. Available domains: {list(DOMAIN_CONFIGURATIONS.keys())}")
    return DOMAIN_CONFIGURATIONS[domain]


def get_all_domains() -> List[str]:
    """Get list of all available domains."""
    return list(DOMAIN_CONFIGURATIONS.keys())


def register_domain_config(config: DomainConfiguration):
    """
    Register a new domain configuration.

    Args:
        config: DomainConfiguration to register
    """
    # Validate configuration
    errors = config.validate()
    if errors:
        raise ValueError(f"Invalid domain configuration: {errors}")

    DOMAIN_CONFIGURATIONS[config.domain] = config


def get_domains_by_expertise(expertise: str) -> List[str]:
    """
    Get domains that require specific expertise.

    Args:
        expertise: Expertise area

    Returns:
        List of domain identifiers
    """
    matching_domains = []
    for domain, config in DOMAIN_CONFIGURATIONS.items():
        if expertise in config.required_expertise:
            matching_domains.append(domain)
    return matching_domains
