"""
Strategy Templates for Custom Decomposition

This module provides predefined templates for creating custom decomposition strategies.
Templates can be customized and used as starting points for domain-specific or
use-case-specific strategies.

Available Templates:
- Domain-Specific: Tailored for specific domains (software, research, data science, etc.)
- Priority-Based: Focus on business value and criticality
- Complexity-Based: Balance cognitive load
- Team-Based: Focus on team capabilities and expertise
"""
from __future__ import annotations


import logging
from typing import Dict, Any, List
from custom_strategy_builder import StrategyConfig

logger = logging.getLogger(__name__)


class StrategyTemplates:
    """Predefined templates for custom strategies."""

    # Domain-specific prompts
    DOMAIN_PROMPTS = {
        "software": """
You are an expert software engineering decomposition specialist. Break down software
development problems into manageable sub-problems considering:
- Architecture and design patterns
- Implementation phases
- Testing and validation
- Documentation and deployment
- Technical debt and refactoring
""",

        "research": """
You are an expert research decomposition specialist. Break down research problems into
manageable sub-problems considering:
- Literature review and background research
- Hypothesis formulation
- Experimental design
- Data collection and analysis
- Validation and peer review
""",

        "data_science": """
You are an expert data science decomposition specialist. Break down data science problems
into manageable sub-problems considering:
- Data acquisition and preprocessing
- Exploratory data analysis
- Feature engineering
- Model development and training
- Validation and deployment
""",

        "business": """
You are an expert business process decomposition specialist. Break down business problems
into manageable sub-problems considering:
- Stakeholder analysis
- Requirements gathering
- Process design
- Implementation planning
- Change management and adoption
""",

        "general": """
You are an expert problem decomposition specialist. Break down the problem into
manageable sub-problems considering:
- Clear objectives and deliverables
- Logical dependencies
- Resource requirements
- Risk mitigation
- Validation and success criteria
"""
    }

    def __init__(self):
        """Initialize templates."""
        self.available_templates = [
            "domain_specific",
            "priority_based",
            "complexity_based",
            "team_based"
        ]
        logger.info(f"StrategyTemplates initialized with {len(self.available_templates)} templates")

    def domain_specific_template(self, domain: str) -> StrategyConfig:
        """
        Template for domain-specific strategy.

        Customizes decomposition for:
        - Domain terminology
        - Domain-specific patterns
        - Common domain approaches

        Args:
            domain: Domain name (software, research, data_science, business, general)

        Returns:
            StrategyConfig configured for domain
        """
        domain = domain.lower()
        if domain not in self.DOMAIN_PROMPTS:
            logger.warning(f"Unknown domain '{domain}', using 'general' template")
            domain = "general"

        prompt = self.DOMAIN_PROMPTS[domain]

        return StrategyConfig(
            strategy_name=f"domain_{domain}",
            description=f"Custom strategy for {domain} domain problems",
            system_prompt=prompt.strip(),
            user_prompt_template="""Break down the following problem into sub-problems appropriate for the {domain} domain:

Problem: {title}
Description: {description}
Domain: {domain}

Consider domain-specific patterns, terminology, and best practices.""",
            decomposition_criteria={
                "focus": "domain_patterns",
                "domain": domain,
                "use_domain_terminology": True
            },
            sub_problem_ordering="sequential",
            add_dependencies=True,
            quality_thresholds={
                "min_sub_problems": 3,
                "max_sub_problems": 7,
                "clarity_score": 0.7,
                "domain_relevance": 0.8
            },
            author="StrategyTemplates",
            version="1.0.0",
            tags=[domain, "domain_specific"]
        )

    def priority_based_template(self) -> StrategyConfig:
        """
        Template for priority-based decomposition.

        Decomposes based on:
        - Business value
        - Criticality
        - Dependencies
        - Risk

        Returns:
            StrategyConfig for priority-based decomposition
        """
        return StrategyConfig(
            strategy_name="priority_custom",
            description="Decompose by business priority and criticality",
            system_prompt="""You are an expert business value decomposition specialist.
Break down problems prioritizing sub-problems by:
1. Business value and impact
2. Criticality and dependencies
3. Risk assessment
4. Resource requirements

Always tackle high-value, low-risk items first.""",
            user_prompt_template="""Break down this problem prioritizing by business value:

Problem: {title}
Description: {description}

Identify and prioritize sub-problems based on their business impact, criticality,
and risk. High-value, low-risk items should come first.""",
            decomposition_criteria={
                "primary_dimension": "business_value",
                "secondary_dimension": "risk",
                "prioritize_mvp": True
            },
            sub_problem_ordering="priority",  # This will order by priority field
            add_dependencies=True,
            dependency_rules=[
                "high_priority_blocks_low_priority",
                "critical_dependencies_first"
            ],
            quality_thresholds={
                "min_priority_coverage": 0.8,
                "value_clarity": 0.7,
                "risk_assessment": 0.6
            },
            author="StrategyTemplates",
            version="1.0.0",
            tags=["priority", "business_value", "risk_based"]
        )

    def complexity_based_template(self) -> StrategyConfig:
        """
        Template for complexity-based decomposition.

        Decomposes to balance cognitive load across sub-problems.

        Returns:
            StrategyConfig for complexity-based decomposition
        """
        return StrategyConfig(
            strategy_name="complexity_custom",
            description="Decompose by balancing complexity across sub-problems",
            system_prompt="""You are an expert complexity balancing specialist.
Break down problems to create sub-problems with:
1. Balanced complexity (target: 5-7/10 each)
2. Manageable cognitive load
3. Clear success criteria
4. Minimal interdependencies

Avoid creating sub-problems that are too simple (<3/10) or too complex (>8/10).""",
            user_prompt_template="""Break down this problem to balance complexity:

Problem: {title}
Description: {description}
Overall Complexity: {complexity}/10

Create sub-problems each with complexity between 5-7/10. Aim for 4-6 sub-problems
with balanced complexity distribution.""",
            decomposition_criteria={
                "target_complexity": 6.0,
                "balance_threshold": 2.0,
                "min_complexity": 5.0,
                "max_complexity": 7.0
            },
            sub_problem_ordering="sequential",
            add_dependencies=True,
            quality_thresholds={
                "complexity_balance": 0.8,
                "complexity_variance": 0.3,
                "cognitive_load": 0.7
            },
            author="StrategyTemplates",
            version="1.0.0",
            tags=["complexity", "balance", "cognitive_load"]
        )

    def team_based_template(self) -> StrategyConfig:
        """
        Template for team-based decomposition.

        Decomposes considering:
        - Team expertise and capabilities
        - Workload distribution
        - Specialization
        - Collaboration patterns

        Returns:
            StrategyConfig for team-based decomposition
        """
        return StrategyConfig(
            strategy_name="team_custom",
            description="Decompose based on team capabilities and expertise",
            system_prompt="""You are an expert team-based decomposition specialist.
Break down problems considering:
1. Team expertise and specialization
2. Workload distribution
3. Collaboration patterns
4. Skill requirements

Match sub-problems to team capabilities and ensure balanced workload.""",
            user_prompt_template="""Break down this problem for team execution:

Problem: {title}
Description: {description}
Available Teams: {teams}

Consider team expertise, workload balance, and collaboration needs when creating
sub-problems. Each sub-problem should be assignable to a specific team based on
their capabilities.""",
            decomposition_criteria={
                "primary_dimension": "team_expertise",
                "workload_balance": True,
                "specialization_match": True
            },
            sub_problem_ordering="sequential",
            add_dependencies=True,
            quality_thresholds={
                "team_utilization": 0.8,
                "expertise_match": 0.7,
                "workload_balance": 0.75
            },
            author="StrategyTemplates",
            version="1.0.0",
            tags=["team", "expertise", "workload"]
        )

    def agile_sprint_template(self) -> StrategyConfig:
        """
        Template for Agile sprint-based decomposition.

        Decomposes for iterative development in sprints.

        Returns:
            StrategyConfig for Agile decomposition
        """
        return StrategyConfig(
            strategy_name="agile_sprint",
            description="Decompose for Agile sprint planning",
            system_prompt="""You are an expert Agile decomposition specialist.
Break down problems into sprint-sized sub-problems:
1. Each sub-problem fits in a 1-2 week sprint
2. Clear acceptance criteria
3. Potentially shippable increments
4. Testable and demonstrable
5. User story format when appropriate

Follow Agile principles and Scrum framework.""",
            user_prompt_template="""Break down this problem for Agile sprint execution:

Problem: {title}
Description: {description}

Create sub-problems that:
- Fit in 1-2 week sprints
- Have clear acceptance criteria
- Deliver user value
- Are testable and demonstrable
- Follow user story format (As a... I want... So that...)""",
            decomposition_criteria={
                "sprint_duration_weeks": 2,
                "user_story_format": True,
                "incremental_delivery": True,
                "potentially_shippable": True
            },
            sub_problem_ordering="priority",  # Highest value first
            add_dependencies=True,
            quality_thresholds={
                "sprint_fit": 0.9,
                "acceptance_criteria_clarity": 0.8,
                "user_value": 0.7
            },
            author="StrategyTemplates",
            version="1.0.0",
            tags=["agile", "scrum", "sprint", "iterative"]
        )

    def research_phased_template(self) -> StrategyConfig:
        """
        Template for research-phase based decomposition.

        Decomposes research problems into standard research phases.

        Returns:
            StrategyConfig for research decomposition
        """
        return StrategyConfig(
            strategy_name="research_phased",
            description="Decompose research problems into standard research phases",
            system_prompt="""You are an expert research methodology specialist.
Break down research problems into standard phases:
1. Literature Review and Background
2. Hypothesis Formulation
3. Research Design
4. Data Collection
5. Analysis and Interpretation
6. Validation and Peer Review
7. Documentation and Publication

Ensure scientific rigor and reproducibility.""",
            user_prompt_template="""Break down this research problem into standard research phases:

Research Problem: {title}
Description: {description}
Domain: {domain}

Follow standard research methodology and ensure each phase has:
- Clear objectives
- Deliverables
- Success criteria
- Validation methods""",
            decomposition_criteria={
                "methodology": "scientific",
                "phases": [
                    "literature_review",
                    "hypothesis",
                    "design",
                    "data_collection",
                    "analysis",
                    "validation",
                    "documentation"
                ],
                "scientific_rigor": True,
                "reproducibility": True
            },
            sub_problem_ordering="sequential",  # Research is typically sequential
            add_dependencies=True,
            quality_thresholds={
                "methodological_soundness": 0.9,
                "reproducibility": 0.8,
                "documentation_quality": 0.8
            },
            author="StrategyTemplates",
            version="1.0.0",
            tags=["research", "scientific", "phased", "methodology"]
        )

    def microservices_template(self) -> StrategyConfig:
        """
        Template for microservices architecture decomposition.

        Decomposes monolithic problems into microservices.

        Returns:
            StrategyConfig for microservices decomposition
        """
        return StrategyConfig(
            strategy_name="microservices_architecture",
            description="Decompose into microservices architecture",
            system_prompt="""You are an expert microservices architecture specialist.
Break down monolithic applications into microservices considering:
1. Domain-driven design and bounded contexts
2. Business capability alignment
3. Data autonomy and ownership
4. API design and contracts
5. Service independence
6. Deployment and scaling
7. Observability and monitoring

Follow microservices best practices and patterns.""",
            user_prompt_template="""Break down this system into microservices:

System: {title}
Description: {description}

Design microservices that:
- Align with business capabilities
- Own their data
- Have clear APIs
- Can be deployed independently
- Follow domain-driven design principles
- Consider data consistency and eventual consistency patterns""",
            decomposition_criteria={
                "architecture_style": "microservices",
                "domain_driven_design": True,
                "bounded_contexts": True,
                "api_first": True,
                "data_autonomy": True
            },
            sub_problem_ordering="priority",  # Core services first
            add_dependencies=True,
            dependency_rules=[
                "core_services_first",
                "api_gateways_last",
                "observability_cross_cutting"
            ],
            quality_thresholds={
                "service_boundaries": 0.8,
                "api_design": 0.8,
                "data_ownership": 0.9
            },
            author="StrategyTemplates",
            version="1.0.0",
            tags=["microservices", "architecture", "ddd", "api"]
        )

    def get_template(self, template_name: str, **kwargs) -> StrategyConfig:
        """
        Get a template by name.

        Args:
            template_name: Name of template
            **kwargs: Additional arguments for template customization

        Returns:
            StrategyConfig for requested template

        Raises:
            ValueError: If template not found
        """
        template_methods = {
            "domain_specific": (self.domain_specific_template, ["domain"]),
            "priority_based": (self.priority_based_template, []),
            "complexity_based": (self.complexity_based_template, []),
            "team_based": (self.team_based_template, []),
            "agile_sprint": (self.agile_sprint_template, []),
            "research_phased": (self.research_phased_template, []),
            "microservices": (self.microservices_template, []),
        }

        if template_name not in template_methods:
            raise ValueError(
                f"Unknown template: {template_name}. "
                f"Available: {list(template_methods.keys())}"
            )

        method, arg_names = template_methods[template_name]

        # Extract required args from kwargs
        args = {k: kwargs[k] for k in arg_names if k in kwargs}

        return method(**args)

    def list_templates(self) -> List[str]:
        """List all available template names."""
        return [
            "domain_specific",
            "priority_based",
            "complexity_based",
            "team_based",
            "agile_sprint",
            "research_phased",
            "microservices"
        ]

    def get_template_info(self, template_name: str) -> Dict[str, Any]:
        """
        Get information about a template.

        Args:
            template_name: Name of template

        Returns:
            Dict with template information
        """
        template_descriptions = {
            "domain_specific": {
                "name": "Domain-Specific",
                "description": "Tailored for specific domains (software, research, data science, etc.)",
                "parameters": ["domain"],
                "use_cases": ["Domain-specific projects", "Specialized workflows", "Industry-specific problems"]
            },
            "priority_based": {
                "name": "Priority-Based",
                "description": "Focus on business value and criticality",
                "parameters": [],
                "use_cases": ["Business projects", "MVP development", "Value-driven development"]
            },
            "complexity_based": {
                "name": "Complexity-Based",
                "description": "Balance cognitive load across sub-problems",
                "parameters": [],
                "use_cases": ["Complex technical problems", "Team balancing", "Cognitive load management"]
            },
            "team_based": {
                "name": "Team-Based",
                "description": "Focus on team capabilities and expertise",
                "parameters": [],
                "use_cases": ["Team-based projects", "Specialized teams", "Collaborative workflows"]
            },
            "agile_sprint": {
                "name": "Agile Sprint",
                "description": "Decompose for Agile sprint planning",
                "parameters": [],
                "use_cases": ["Scrum projects", "Iterative development", "Sprint planning"]
            },
            "research_phased": {
                "name": "Research Phased",
                "description": "Decompose research problems into standard phases",
                "parameters": [],
                "use_cases": ["Academic research", "Scientific studies", "PhD projects"]
            },
            "microservices": {
                "name": "Microservices",
                "description": "Decompose into microservices architecture",
                "parameters": [],
                "use_cases": ["Cloud architecture", "DDD projects", "Service design"]
            }
        }

        return template_descriptions.get(template_name, {
            "name": template_name,
            "description": "Unknown template",
            "parameters": [],
            "use_cases": []
        })
