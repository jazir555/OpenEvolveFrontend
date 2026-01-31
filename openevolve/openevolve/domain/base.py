"""
Domain Optimizer Base Class
Provides the foundation for all domain-specific optimizers

Author: AI Architecture Team
Date: 2026-01-30
"""

from typing import Dict, Any, Optional, List
from ..unified.config import UnifiedEvolutionConfig


class DomainOptimizer:
    """
    Base class for domain-specific optimizers

    Provides:
    - Domain-specific configuration
    - Recommended system/mode selection
    - Domain-specific evaluation metrics
    - Sub-domain specializations
    """

    domain_name: str = "general"

    def __init__(self, sub_domain: str = "general"):
        """
        Initialize domain optimizer

        Args:
            sub_domain: Sub-domain specialization
        """
        self.sub_domain = sub_domain
        self.sub_domain_configs = {}
        self.config = self.get_default_config()

    def get_default_config(self) -> UnifiedEvolutionConfig:
        """
        Get default configuration for this domain

        Returns:
            UnifiedEvolutionConfig with domain-specific settings
        """
        return UnifiedEvolutionConfig()

    def get_recommended_system(self) -> str:
        """
        Get recommended evolutionary system

        Returns:
            'openevolve' or 'loongflow'
        """
        return "openevolve"

    def get_recommended_mode(self) -> str:
        """
        Get recommended evolutionary mode

        Returns:
            One of: 'pes', 'qd', 'mo', 'adversarial', 'standard'
        """
        return "standard"

    def get_domain_metrics(self) -> List[str]:
        """
        Get domain-specific metric names

        Returns:
            List of metric names (e.g., ['sharpe_ratio', 'max_drawdown'])
        """
        return ["fitness"]

    def evaluate_solution(
        self,
        solution: str,
        problem: str,
        constraints: Optional[Dict[str, Any]] = None
    ) -> Dict[str, float]:
        """
        Domain-specific evaluation metrics

        Args:
            solution: Solution code/text
            problem: Problem description
            constraints: Additional constraints

        Returns:
            Dictionary of metric names to values

        Example:
            >>> metrics = optimizer.evaluate_solution(portfolio_code, problem)
            >>> print(metrics['sharpe_ratio'])
        """
        # Base implementation - override in subclasses
        return {"fitness": 0.5}

    async def optimize(
        self,
        problem: str,
        constraints: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Run optimization with domain-specific configuration

        Args:
            problem: Problem description
            constraints: Additional constraints
            **kwargs: Additional parameters

        Returns:
            Optimization result with domain-specific metrics

        Example:
            >>> result = await optimizer.optimize("Maximize return")
            >>> print(result['best_solution'])
            >>> print(result['sharpe_ratio'])
        """
        # Import here to avoid circular dependency
        from ..unified.api import evolve

        # Run evolution with domain config
        result = await evolve(
            problem_statement=problem,
            config=self.config,
            constraints=constraints,
            **kwargs
        )

        # Add domain-specific evaluation
        if result.get('best_solution'):
            domain_metrics = self.evaluate_solution(
                result['best_solution'],
                problem,
                constraints
            )
            result['domain_metrics'] = domain_metrics

        # Add metadata
        result['domain'] = self.domain_name
        result['sub_domain'] = self.sub_domain
        result['recommended_system'] = self.get_recommended_system()
        result['recommended_mode'] = self.get_recommended_mode()

        return result

    def get_sub_domain_config(self, sub_domain: str) -> UnifiedEvolutionConfig:
        """
        Get configuration for specific sub-domain

        Args:
            sub_domain: Sub-domain name

        Returns:
            Configuration for that sub-domain
        """
        if sub_domain in self.sub_domain_configs:
            return self.sub_domain_configs[sub_domain]
        return self.get_default_config()

    def list_sub_domains(self) -> List[str]:
        """
        List available sub-domains

        Returns:
            List of sub-domain names
        """
        return list(self.sub_domain_configs.keys())
