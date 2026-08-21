"""
Web Design Domain Optimizer
Specialized optimizer for web design problems

Problems:
- Landing page optimization (conversion rate)
- UX optimization (user engagement)
- A/B testing (variant generation)

Best System: OpenEvolve (Standard mode)
Why: Fast evaluations (seconds), well-understood domain

Configuration:
- Evaluation cost: "cheap"
- Mode: Standard
- Population size: 50
- Generations: 20
- Mutation rate: 0.3
- Crossover rate: 0.7

Metrics:
- Conversion rate
- Bounce rate
- Time on page
- User satisfaction

Author: AI Architecture Team
Date: 2026-01-30
"""

from typing import Dict, Any, Optional, List
from ..unified.config import UnifiedEvolutionConfig, EvolutionMode, DomainType, MOConfig, LLMConfig, EvaluatorConfig, DatabaseConfig
from .base import DomainOptimizer
from .heuristics import clamp, saturating, signal_coverage


class WebDesignOptimizer(DomainOptimizer):
    """
    Web design domain optimizer

    Specialized for:
    - Landing page optimization
    - UX optimization
    - A/B testing

    Example:
        >>> optimizer = WebDesignOptimizer(sub_domain="landing_page")
        >>> result = await optimizer.optimize(
        ...     "Optimize landing page for maximum conversion",
        ...     constraints={"max_load_time": 3.0, "mobile_first": True}
        ... )
        >>> print(result['domain_metrics']['conversion_rate'])
    """

    domain_name = "web_design"

    # Signals used by the deterministic metric calculations
    ACCESSIBILITY_SIGNALS = [
        "aria-", "role=", "lang=", "<main", "<nav", "<header", "<footer",
        "tabindex", "for=", "skip to content"
    ]
    SEO_SIGNALS = [
        "<title", "name=\"description\"", "<h1", "canonical", "og:",
        "application/ld+json", "sitemap", "robots"
    ]
    TRUST_SIGNALS = [
        "testimonial", "review", "rating", "guarantee", "secure",
        "privacy", "customers", "trusted", "money-back", "certified"
    ]
    ENGAGEMENT_SIGNALS = [
        "<video", "<picture", "<svg", "carousel", "accordion", "faq",
        "animation", "interactive", "<section", "<article"
    ]

    def __init__(self, sub_domain: str = "general"):
        """
        Initialize web design optimizer

        Args:
            sub_domain: One of 'general', 'landing_page', 'ux', 'ab_testing'
        """
        super().__init__(sub_domain)

        # Define sub-domain configurations
        self.sub_domain_configs = {
            "general": self._general_config(),
            "landing_page": self._landing_page_config(),
            "ux": self._ux_config(),
            "ab_testing": self._ab_testing_config()
        }

        # Set active config
        self.config = self.sub_domain_configs.get(
            sub_domain,
            self._general_config()
        )

    def get_recommended_system(self) -> str:
        """OpenEvolve standard for fast iterations"""
        return "openevolve"

    def get_recommended_mode(self) -> str:
        """Standard mode for rapid prototyping"""
        return "standard"

    def get_domain_metrics(self) -> List[str]:
        """Web design-specific metrics"""
        return [
            "conversion_rate",
            "bounce_rate",
            "time_on_page",
            "user_satisfaction",
            "click_through_rate",
            "scroll_depth",
            "form_completion_rate",
            "load_time",
            "accessibility_score",
            "seo_score"
        ]

    def get_default_config(self) -> UnifiedEvolutionConfig:
        """Get default web design configuration"""
        return self._general_config()

    # ========================================================================
    # SUB-DOMAIN CONFIGURATIONS
    # ========================================================================

    def _general_config(self) -> UnifiedEvolutionConfig:
        """
        General web design configuration

        Uses standard mode for rapid iteration
        """
        return UnifiedEvolutionConfig(
            # Domain
            domain=DomainType.WEB_DESIGN,

            # Evolution mode
            evolution_mode=EvolutionMode.STANDARD,

            # LLM configuration
            llm=LLMConfig(
                temperature=0.7,
                timeout=60,  # Fast evaluation
                retries=3
            ),

            # Evaluation (fast - seconds)
            max_iterations=100,
            evaluator=EvaluatorConfig(
                timeout=30,  # 30 seconds per variant
                max_retries=3,
                early_stopping=True,
                early_stopping_patience=10,
                parallel_evaluations=20  # High parallelism
            ),

            # Population for diversity
            database=DatabaseConfig(
                population_size=50,
                archive_size=20
            )
        )

    def _landing_page_config(self) -> UnifiedEvolutionConfig:
        """
        Landing page optimization configuration

        Focus on conversion rate
        """
        config = self._general_config()

        # More iterations for convergence
        config.max_iterations = 150

        # Larger population for more variants
        config.database.population_size = 75

        # Slightly higher temperature for creative copy
        config.llm.temperature = 0.8

        return config

    def _ux_config(self) -> UnifiedEvolutionConfig:
        """
        UX optimization configuration

        Focus on user engagement
        """
        config = self._general_config()

        # Multi-objective: engagement, satisfaction, accessibility
        config.mo = MOConfig(
            enabled=True,
            objectives=["engagement", "satisfaction", "accessibility"],
            algorithm="nsga2",
            pareto_size=30
        )

        # Moderate iterations
        config.max_iterations = 120

        return config

    def _ab_testing_config(self) -> UnifiedEvolutionConfig:
        """
        A/B testing configuration

        Focus on variant generation
        """
        config = self._general_config()

        # Fewer iterations (rapid testing)
        config.max_iterations = 50

        # High mutation rate for diverse variants
        config.llm.temperature = 0.9

        # Smaller population
        config.database.population_size = 30

        return config

    # ========================================================================
    # DOMAIN-SPECIFIC EVALUATION
    # ========================================================================

    def evaluate_solution(
        self,
        solution: str,
        problem: str,
        constraints: Optional[Dict[str, Any]] = None
    ) -> Dict[str, float]:
        """
        Evaluate web design

        Args:
            solution: HTML/CSS/JS code or design spec
            problem: Problem description
            constraints: Constraints (max_load_time, mobile_first, etc.)

        Returns:
            Dictionary of web metrics

        Example:
            >>> metrics = optimizer.evaluate_solution(
            ...     landing_page_html,
            ...     "Optimize for conversion",
            ...     {"max_load_time": 3.0}
            ... )
            >>> print(metrics['conversion_rate'])
        """
        # Parse design
        design = self._parse_design(solution)

        # Calculate metrics
        metrics = self._calculate_web_metrics(
            design,
            problem,
            constraints
        )

        return metrics

    def _parse_design(self, solution: str) -> Dict[str, Any]:
        """
        Parse web design from solution

        Args:
            solution: HTML/CSS/JS code

        Returns:
            Design components (including the raw markup for scoring)
        """
        import re

        design: Dict[str, Any] = {
            "source": solution or "",
            "html_elements": [],
            "css_rules": 0,
            "javascript_functions": 0,
            "images": 0,
            "forms": 0,
            "calls_to_action": 0
        }

        # Count HTML elements
        design["html_elements"] = re.findall(r'<(\w+)', solution)

        # Count CSS rules
        design["css_rules"] = len(re.findall(r'\{[^}]*\}', solution))

        # Count JavaScript functions
        design["javascript_functions"] = len(re.findall(r'function\s+\w+', solution))

        # Count images
        design["images"] = len(re.findall(r'<img', solution))

        # Count forms
        design["forms"] = len(re.findall(r'<form', solution))

        # Count CTAs
        cta_patterns = [r'button', r'submit', r'click', r'register', r'sign up']
        for pattern in cta_patterns:
            matches = re.findall(pattern, solution, re.IGNORECASE)
            design["calls_to_action"] += len(matches)

        # Structural details used by the metric calculations
        design["headings"] = len(re.findall(r'<h[1-6]\b', solution, re.IGNORECASE))
        design["inputs"] = len(re.findall(r'<(input|select|textarea)\b', solution, re.IGNORECASE))
        design["labels"] = len(re.findall(r'<label\b', solution, re.IGNORECASE))
        design["images_with_alt"] = len(re.findall(r'<img[^>]*\balt\s*=', solution, re.IGNORECASE))
        design["inline_scripts"] = len(re.findall(r'<script(?![^>]*\bsrc=)', solution, re.IGNORECASE))
        design["external_assets"] = len(
            re.findall(r'<(script[^>]*\bsrc=|link[^>]*\bhref=)', solution, re.IGNORECASE)
        )
        design["word_count"] = len(re.findall(r'\b[A-Za-z]{2,}\b', re.sub(r'<[^>]+>', ' ', solution)))

        return design

    def _calculate_web_metrics(
        self,
        design: Dict[str, Any],
        problem: str,
        constraints: Optional[Dict[str, Any]] = None
    ) -> Dict[str, float]:
        """
        Calculate web metrics from measurable properties of the markup

        Every metric is a deterministic function of signals found in the
        candidate (semantic structure, accessibility attributes, CTA placement,
        payload weight, SEO metadata, form friction), so evolution gets a real
        gradient without A/B testing platforms, analytics or Lighthouse.

        Args:
            design: Design components from :meth:`_parse_design`
            problem: Problem description
            constraints: Constraints (max_load_time, min_accessibility, ...)

        Returns:
            Dictionary of metrics (normalized 0-1, higher is better unless noted)
        """
        source = design.get("source", "")
        elements = max(1, len(design.get("html_elements", [])))

        # --- Payload / performance -----------------------------------------
        payload_kb = len(source.encode("utf-8")) / 1024.0
        weight_penalty = saturating(payload_kb, 250.0)
        asset_penalty = saturating(design.get("external_assets", 0), 20)
        script_penalty = saturating(design.get("inline_scripts", 0), 6)
        load_time = clamp(
            1.0 - 0.5 * weight_penalty - 0.3 * asset_penalty - 0.2 * script_penalty
        )

        # --- Accessibility --------------------------------------------------
        images = design.get("images", 0)
        alt_ratio = 1.0 if images == 0 else clamp(design.get("images_with_alt", 0) / images)
        inputs = design.get("inputs", 0)
        label_ratio = 1.0 if inputs == 0 else clamp(design.get("labels", 0) / inputs)
        accessibility_score = clamp(
            0.35 * alt_ratio
            + 0.25 * label_ratio
            + 0.25 * signal_coverage(source, self.ACCESSIBILITY_SIGNALS)
            + 0.15 * saturating(design.get("headings", 0), 3)
        )

        # --- SEO ------------------------------------------------------------
        seo_score = clamp(
            0.5 * signal_coverage(source, self.SEO_SIGNALS)
            + 0.25 * saturating(design.get("headings", 0), 4)
            + 0.25 * saturating(design.get("word_count", 0), 300)
        )

        # --- Engagement / conversion ---------------------------------------
        cta_density = saturating(design.get("calls_to_action", 0), 6)
        trust_signals = signal_coverage(source, self.TRUST_SIGNALS)
        engagement_signals = signal_coverage(source, self.ENGAGEMENT_SIGNALS)
        content_depth = saturating(design.get("word_count", 0), 400)
        form_friction = saturating(inputs, 8)

        conversion_rate = clamp(
            0.01
            + 0.08 * cta_density
            + 0.05 * trust_signals
            + 0.03 * load_time
            + 0.02 * accessibility_score
            - 0.04 * form_friction,
            0.0,
            0.25,
        )

        click_through_rate = clamp(
            0.01 + 0.10 * cta_density + 0.04 * saturating(design.get("calls_to_action", 0), elements),
            0.0,
            0.3,
        )

        # Bounce rate is a cost: lower is better
        bounce_rate = clamp(
            0.75
            - 0.20 * load_time
            - 0.15 * content_depth
            - 0.15 * engagement_signals
            - 0.10 * accessibility_score,
            0.05,
            0.95,
        )

        time_on_page = clamp(
            0.25 * content_depth
            + 0.25 * engagement_signals
            + 0.25 * saturating(design.get("headings", 0), 5)
            + 0.25 * (1.0 - bounce_rate)
        )

        scroll_depth = clamp(
            0.4 * saturating(design.get("headings", 0), 6)
            + 0.3 * content_depth
            + 0.3 * (1.0 - bounce_rate)
        )

        form_completion_rate = clamp(
            0.3 + 0.35 * label_ratio + 0.2 * (1.0 - form_friction) + 0.15 * trust_signals
        ) if inputs else 0.0

        user_satisfaction = clamp(
            0.3 * load_time
            + 0.25 * accessibility_score
            + 0.2 * (1.0 - bounce_rate)
            + 0.15 * engagement_signals
            + 0.1 * seo_score
        )

        metrics = {
            "conversion_rate": conversion_rate,
            "bounce_rate": bounce_rate,              # lower is better
            "time_on_page": time_on_page,
            "user_satisfaction": user_satisfaction,
            "click_through_rate": click_through_rate,
            "scroll_depth": scroll_depth,
            "form_completion_rate": form_completion_rate,
            "load_time": load_time,                  # pagespeed-style score
            "accessibility_score": accessibility_score,
            "seo_score": seo_score,
        }

        # Constraints tighten the performance target rather than being ignored
        if constraints and "max_load_time" in constraints:
            budget = max(0.5, float(constraints["max_load_time"]))
            estimated_seconds = 0.5 + 9.5 * (1.0 - load_time)
            if estimated_seconds > budget:
                metrics["load_time"] = clamp(load_time * budget / estimated_seconds)

        return metrics

    # ========================================================================
    # UTILITY METHODS
    # ========================================================================

    def get_web_constraints(
        self,
        max_load_time: float = 3.0,
        mobile_first: bool = True,
        min_accessibility: float = 0.8,
        min_seo: float = 0.7
    ) -> Dict[str, Any]:
        """
        Get standard web constraints

        Args:
            max_load_time: Maximum load time (seconds)
            mobile_first: Mobile-first design
            min_accessibility: Minimum accessibility score (0-1)
            min_seo: Minimum SEO score (0-1)

        Returns:
            Constraints dictionary

        Example:
            >>> constraints = optimizer.get_web_constraints(
            ...     max_load_time=2.0,
            ...     min_accessibility=0.9
            ... )
        """
        return {
            "max_load_time": max_load_time,
            "mobile_first": mobile_first,
            "min_accessibility": min_accessibility,
            "min_seo": min_seo
        }

    def validate_design(
        self,
        metrics: Dict[str, float],
        constraints: Optional[Dict[str, Any]] = None
    ) -> tuple[bool, List[str]]:
        """
        Validate web design

        Args:
            metrics: Web metrics
            constraints: Constraints

        Returns:
            (is_valid, list_of_violations)

        Example:
            >>> is_valid, violations = optimizer.validate_design(
            ...     metrics,
            ...     {"max_load_time": 3.0}
            ... )
        """
        if constraints is None:
            return True, []

        violations = []

        # Check load time
        if "max_load_time" in constraints:
            # Load time metric is inverted (higher = better)
            # So we check: 1 - load_time_score
            load_time_normalized = 1 - metrics.get("load_time", 0)
            max_load_time_normalized = constraints["max_load_time"] / 10.0  # Normalize to 0-1
            if load_time_normalized > max_load_time_normalized:
                violations.append(
                    f"Load time exceeds maximum: {load_time_normalized * 10:.2f}s > {constraints['max_load_time']:.1f}s"
                )

        # Check accessibility
        if "min_accessibility" in constraints:
            if metrics.get("accessibility_score", 0) < constraints["min_accessibility"]:
                violations.append(
                    f"Accessibility score too low: {metrics['accessibility_score']:.2%} < {constraints['min_accessibility']:.2%}"
                )

        # Check SEO
        if "min_seo" in constraints:
            if metrics.get("seo_score", 0) < constraints["min_seo"]:
                violations.append(
                    f"SEO score too low: {metrics['seo_score']:.2%} < {constraints['min_seo']:.2%}"
                )

        return len(violations) == 0, violations

    def generate_ab_test_variants(
        self,
        base_design: str,
        num_variants: int = 5
    ) -> List[str]:
        """
        Generate A/B test variants

        Applies deterministic, single-factor mutations to the base design so each
        variant differs in exactly one testable dimension (CTA copy, urgency,
        social proof, form friction, above-the-fold emphasis).

        Args:
            base_design: Base HTML/CSS/JS
            num_variants: Number of variants to generate

        Returns:
            List of variant designs

        Example:
            >>> variants = optimizer.generate_ab_test_variants(
            ...     base_html,
            ...     num_variants=10
            ... )
        """
        import re

        mutations = [
            # (name, callable applying the mutation)
            ("cta_copy", lambda html: re.sub(
                r'(<button[^>]*>)([^<]*)(</button>)',
                r'\1Get Started Free\3',
                html,
                count=1,
                flags=re.IGNORECASE,
            )),
            ("urgency", lambda html: html.replace(
                "</h1>", "</h1>\n<p class=\"urgency\">Limited time offer</p>", 1
            )),
            ("social_proof", lambda html: html.replace(
                "</form>",
                "</form>\n<p class=\"testimonial\">Trusted by 10,000 customers</p>",
                1,
            )),
            ("reduced_friction", lambda html: re.sub(
                r'<input(?![^>]*type="(?:submit|hidden)")[^>]*>', '', html, count=1,
                flags=re.IGNORECASE,
            )),
            ("above_the_fold", lambda html: re.sub(
                r'(<h1[^>]*>)', r'\1[New] ', html, count=1, flags=re.IGNORECASE
            )),
            ("accessibility", lambda html: re.sub(
                r'<img(?![^>]*\balt=)', '<img alt="product illustration"', html,
                count=1, flags=re.IGNORECASE,
            )),
        ]

        variants: List[str] = []
        for index in range(num_variants):
            name, mutate = mutations[index % len(mutations)]
            variant = mutate(base_design)
            if variant == base_design:
                # Mutation did not apply: annotate the variant so it stays distinct
                variant = f"{base_design}\n<!-- variant: {name}-{index} -->"
            variants.append(variant)

        return variants

    def suggest_improvements(
        self,
        metrics: Dict[str, float],
        target_metrics: Dict[str, float]
    ) -> List[str]:
        """
        Suggest design improvements

        Args:
            metrics: Current metrics
            target_metrics: Target metrics

        Returns:
            List of improvement suggestions

        Example:
            >>> suggestions = optimizer.suggest_improvements(
            ...     metrics,
            ...     {"conversion_rate": 0.08}
            ... )
        """
        suggestions = []

        # Compare current to target
        for metric, target in target_metrics.items():
            current = metrics.get(metric, 0)
            if current < target:
                if metric == "conversion_rate":
                    suggestions.append("Add clearer call-to-action above the fold")
                    suggestions.append("Simplify form to reduce friction")
                    suggestions.append("Add social proof (testimonials, reviews)")
                elif metric == "bounce_rate":
                    suggestions.append("Improve page load speed")
                    suggestions.append("Match ad copy to landing page headline")
                    suggestions.append("Add engaging visual content")
                elif metric == "time_on_page":
                    suggestions.append("Add more compelling content")
                    suggestions.append("Include video or interactive elements")
                    suggestions.append("Improve content structure and readability")
                elif metric == "accessibility_score":
                    suggestions.append("Add alt text to images")
                    suggestions.append("Ensure sufficient color contrast")
                    suggestions.append("Make site keyboard navigable")
                elif metric == "load_time":
                    suggestions.append("Optimize image sizes")
                    suggestions.append("Minimize CSS and JavaScript")
                    suggestions.append("Enable browser caching")

        return suggestions
