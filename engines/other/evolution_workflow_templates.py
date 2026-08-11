"""
Evolution and Adversarial Workflow Templates for BubbleLabs

Provides pre-configured workflow templates for common evolution and
adversarial testing scenarios. These templates can be loaded and customized.

Author: OpenEvolve Frontend Team
"""


from typing import Dict, Any, List
from dataclasses import dataclass, asdict


@dataclass
class WorkflowTemplate:
    """Template for evolution/adversarial workflows"""
    name: str
    description: str
    category: str  # "evolution" or "adversarial"
    config: Dict[str, Any]
    example_content: str
    use_cases: List[str]


# =============================================================================
# EVOLUTION TEMPLATES
# =============================================================================

EVOLUTION_TEMPLATES = {
    "code_optimization": WorkflowTemplate(
        name="Code Optimization",
        description="Evolve code for improved performance, readability, and structure",
        category="evolution",
        config={
            "max_generations": 100,
            "population_size": 20,
            "mutation_rate": 0.1,
            "crossover_rate": 0.7,
            "fitness_type": "custom",
            "enable_maker_voting": True,
            "voting_threshold": 3,
            "selection_method": "tournament",
            "tournament_size": 3,
            "elitism_count": 2
        },
        example_content="""
def example_function(data):
    # Process data
    result = []
    for item in data:
        if item > 0:
            result.append(item * 2)
    return result
""",
        use_cases=[
            "Optimize existing code for better performance",
            "Improve code structure and readability",
            "Reduce code complexity while maintaining functionality"
        ]
    ),

    "prompt_refinement": WorkflowTemplate(
        name="Prompt Refinement",
        description="Evolve prompts for better LLM responses and task completion",
        category="evolution",
        config={
            "max_generations": 50,
            "population_size": 15,
            "mutation_rate": 0.15,
            "crossover_rate": 0.8,
            "fitness_type": "custom",
            "enable_maker_voting": True,
            "voting_threshold": 2,
            "enable_mdap_decomposition": True,
            "decomposition_depth": 2
        },
        example_content="Write a function that sorts a list of numbers in ascending order.",
        use_cases=[
            "Improve prompt clarity and specificity",
            "Add helpful examples and context to prompts",
            "Optimize prompts for specific models"
        ]
    ),

    "text_summarization": WorkflowTemplate(
        name="Text Summarization Evolution",
        description="Evolve text summaries for conciseness and completeness",
        category="evolution",
        config={
            "max_generations": 75,
            "population_size": 25,
            "mutation_rate": 0.12,
            "crossover_rate": 0.75,
            "fitness_type": "custom",
            "enable_maker_voting": False,
            "selection_method": "roulette"
        },
        example_content="This is a long text that needs to be summarized...",
        use_cases=[
            "Create concise summaries of long documents",
            "Extract key information from texts",
            "Generate abstracts for papers"
        ]
    ),

    "api_response_optimization": WorkflowTemplate(
        name="API Response Optimization",
        description="Evolve API responses for efficiency and clarity",
        category="evolution",
        config={
            "max_generations": 60,
            "population_size": 18,
            "mutation_rate": 0.08,
            "crossover_rate": 0.7,
            "fitness_type": "custom",
            "enable_maker_voting": True,
            "voting_threshold": 3
        },
        example_content='{"status": "success", "data": {...}}',
        use_cases=[
            "Optimize JSON response structures",
            "Reduce payload sizes",
            "Improve response clarity"
        ]
    ),

    "maker_voting_evolution": WorkflowTemplate(
        name="MAKER Voting Evolution",
        description="Evolution with MAKER first-to-ahead-by-k voting for zero-error results",
        category="evolution",
        config={
            "max_generations": 120,
            "population_size": 30,
            "mutation_rate": 0.1,
            "crossover_rate": 0.7,
            "fitness_type": "custom",
            "enable_maker_voting": True,
            "voting_threshold": 3,
            "adaptive_voting": True,
            "diversity_threshold": 0.3
        },
        example_content="# Your code or content to evolve",
        use_cases=[
            "High-stakes evolution requiring zero errors",
            "Tasks requiring consensus on best solutions",
            "Multi-criteria optimization problems"
        ]
    ),

    "mdap_decomposition_evolution": WorkflowTemplate(
        name="MDAP Decomposition Evolution",
        description="Evolution with task decomposition for complex problems",
        category="evolution",
        config={
            "max_generations": 150,
            "population_size": 25,
            "mutation_rate": 0.1,
            "crossover_rate": 0.7,
            "fitness_type": "custom",
            "enable_mdap_decomposition": True,
            "decomposition_depth": 5,
            "max_subtasks": 15
        },
        example_content="# Complex problem requiring decomposition",
        use_cases=[
            "Large-scale refactoring projects",
            "Complex multi-objective optimization",
            "Tasks with multiple interdependent components"
        ]
    )
}


# =============================================================================
# ADVERSARIAL TEMPLATES
# =============================================================================

ADVERSARIAL_TEMPLATES = {
    "security_audit": WorkflowTemplate(
        name="Security Audit",
        description="Comprehensive security vulnerability assessment",
        category="adversarial",
        config={
            "adversarial_rounds": 5,
            "attack_strength": 0.7,
            "red_team_size": 5,
            "blue_team_size": 3,
            "coevolution": True,
            "enable_maker_voting": True,
            "voting_threshold": 3,
            "enable_mdap_defense": True
        },
        example_content="""
def authenticate(username, password):
    # Check credentials
    if username == "admin" and password == "password123":
        return True
    return False
""",
        use_cases=[
            "Find security vulnerabilities in code",
            "Test authentication and authorization systems",
            "Identify injection vulnerabilities"
        ]
    ),

    "prompt_injection_testing": WorkflowTemplate(
        name="Prompt Injection Testing",
        description="Test prompts against adversarial injection attacks",
        category="adversarial",
        config={
            "adversarial_rounds": 7,
            "attack_strength": 0.8,
            "red_team_size": 4,
            "blue_team_size": 4,
            "coevolution": True,
            "attack_diversity": True
        },
        example_content="You are a helpful assistant. Provide information about...",
        use_cases=[
            "Test LLM prompt robustness",
            "Identify jailbreak vulnerabilities",
            "Improve prompt security"
        ]
    ),

    "code_robustness": WorkflowTemplate(
        name="Code Robustness Testing",
        description="Test code with edge cases and invalid inputs",
        category="adversarial",
        config={
            "adversarial_rounds": 5,
            "attack_strength": 0.5,
            "red_team_size": 3,
            "blue_team_size": 3,
            "coevolution": False,
            "ensemble_defense": True
        },
        example_content="""
def divide_numbers(a, b):
    return a / b
""",
        use_cases=[
            "Test error handling",
            "Find edge case failures",
            "Improve input validation"
        ]
    ),

    "maker_red_team": WorkflowTemplate(
        name="MAKER Red Team Assessment",
        description="Red team testing with MAKER voting for reliable attack generation",
        category="adversarial",
        config={
            "adversarial_rounds": 5,
            "attack_strength": 0.6,
            "red_team_size": 5,
            "blue_team_size": 3,
            "enable_maker_voting": True,
            "voting_threshold": 3,
            "attack_decomposition": True
        },
        example_content="# Your code to test",
        use_cases=[
            "High-confidence vulnerability identification",
            "Zero-error attack generation",
            "Comprehensive security assessment"
        ]
    ),

    "mdap_blue_team": WorkflowTemplate(
        name="MDAP Blue Team Defense",
        description="Defense generation with maximal task decomposition",
        category="adversarial",
        config={
            "adversarial_rounds": 5,
            "attack_strength": 0.5,
            "defense_strength": 1.0,
            "red_team_size": 3,
            "blue_team_size": 5,
            "enable_mdap_defense": True,
            "max_defenses": 15,
            "defense_layering": True
        },
        example_content="# Your code to protect",
        use_cases=[
            "Generate comprehensive defense strategies",
            "Layered security implementation",
            "Thorough vulnerability coverage"
        ]
    ),

    "coevolution_hardening": WorkflowTemplate(
        name="Attack-Defense Coevolution",
        description="Simultaneous evolution of attacks and defenses",
        category="adversarial",
        config={
            "adversarial_rounds": 10,
            "attack_strength": 0.6,
            "defense_strength": 0.8,
            "red_team_size": 4,
            "blue_team_size": 4,
            "coevolution": True,
            "enable_maker_voting": True,
            "enable_mdap_defense": True
        },
        example_content="# System to harden through coevolution",
        use_cases=[
            "Adversarial training",
            "System hardening",
            "Robustness improvement"
        ]
    )
}


# =============================================================================
# TEMPLATE MANAGEMENT
# =============================================================================

class TemplateManager:
    """Manager for evolution and adversarial workflow templates"""

    def __init__(self):
        self.evolution_templates = EVOLUTION_TEMPLATES
        self.adversarial_templates = ADVERSARIAL_TEMPLATES

    def get_all_templates(self) -> Dict[str, WorkflowTemplate]:
        """Get all available templates"""
        return {**self.evolution_templates, **self.adversarial_templates}

    def get_evolution_templates(self) -> Dict[str, WorkflowTemplate]:
        """Get evolution templates only"""
        return self.evolution_templates

    def get_adversarial_templates(self) -> Dict[str, WorkflowTemplate]:
        """Get adversarial templates only"""
        return self.adversarial_templates

    def get_template(self, template_id: str) -> WorkflowTemplate:
        """Get specific template by ID"""
        all_templates = self.get_all_templates()
        return all_templates.get(template_id)

    def get_templates_by_category(self, category: str) -> Dict[str, WorkflowTemplate]:
        """Get templates filtered by category"""
        if category == "evolution":
            return self.evolution_templates
        elif category == "adversarial":
            return self.adversarial_templates
        return {}

    def search_templates(self, query: str) -> List[WorkflowTemplate]:
        """Search templates by name, description, or use cases"""
        query = query.lower()
        results = []

        for template in self.get_all_templates().values():
            if (query in template.name.lower() or
                query in template.description.lower() or
                any(query in use_case.lower() for use_case in template.use_cases)):
                results.append(template)

        return results

    def create_custom_template(
        self,
        name: str,
        description: str,
        category: str,
        config: Dict[str, Any],
        example_content: str,
        use_cases: List[str]
    ) -> WorkflowTemplate:
        """Create a custom template"""
        return WorkflowTemplate(
            name=name,
            description=description,
            category=category,
            config=config,
            example_content=example_content,
            use_cases=use_cases
        )


# =============================================================================
# TEMPLATE EXPORT FUNCTIONS
# =============================================================================

def export_template_to_dict(template: WorkflowTemplate) -> Dict[str, Any]:
    """Export template to dictionary format"""
    return asdict(template)


def import_template_from_dict(data: Dict[str, Any]) -> WorkflowTemplate:
    """Import template from dictionary format"""
    return WorkflowTemplate(**data)


def get_template_recommended_params(template: WorkflowTemplate) -> str:
    """Get formatted description of recommended parameters"""
    lines = [
        f"### {template.name}",
        f"",
        f"**Description:** {template.description}",
        f"",
        f"**Configuration:**"
    ]

    for key, value in template.config.items():
        lines.append(f"- {key}: {value}")

    lines.append("")
    lines.append("**Use Cases:**")

    for i, use_case in enumerate(template.use_cases, 1):
        lines.append(f"{i}. {use_case}")

    return "\n".join(lines)


# =============================================================================
# PRESET CONFIGURATIONS
# =============================================================================

EVOLUTION_PRESETS = {
    "fast_exploration": {
        "population_size": 10,
        "max_generations": 30,
        "mutation_rate": 0.15,
        "crossover_rate": 0.8
    },
    "balanced": {
        "population_size": 20,
        "max_generations": 100,
        "mutation_rate": 0.1,
        "crossover_rate": 0.7
    },
    "deep_search": {
        "population_size": 50,
        "max_generations": 200,
        "mutation_rate": 0.05,
        "crossover_rate": 0.6
    },
    "maker_voting": {
        "population_size": 25,
        "max_generations": 120,
        "enable_maker_voting": True,
        "voting_threshold": 3,
        "adaptive_voting": True
    }
}

ADVERSARIAL_PRESETS = {
    "quick_test": {
        "adversarial_rounds": 3,
        "red_team_size": 2,
        "blue_team_size": 2,
        "attack_strength": 0.5
    },
    "standard": {
        "adversarial_rounds": 5,
        "red_team_size": 3,
        "blue_team_size": 3,
        "attack_strength": 0.6
    },
    "comprehensive": {
        "adversarial_rounds": 10,
        "red_team_size": 5,
        "blue_team_size": 5,
        "attack_strength": 0.7,
        "coevolution": True
    },
    "maker_enhanced": {
        "adversarial_rounds": 5,
        "red_team_size": 5,
        "blue_team_size": 3,
        "enable_maker_voting": True,
        "voting_threshold": 3,
        "attack_decomposition": True
    }
}


def get_preset(preset_name: str, category: str) -> Dict[str, Any]:
    """Get preset configuration by name and category"""
    if category == "evolution":
        return EVOLUTION_PRESETS.get(preset_name, {})
    elif category == "adversarial":
        return ADVERSARIAL_PRESETS.get(preset_name, {})
    return {}
