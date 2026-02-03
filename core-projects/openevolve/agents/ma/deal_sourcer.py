"""
M&A Deal Sourcer Agent

Continuously scans market for potential acquisition targets,
screens companies, and identifies strategic fit.
"""

import asyncio
import logging
from typing import Optional, List, Dict, Any
from datetime import datetime
from dataclasses import dataclass

from openevolve.agents.ma.schemas import (
    Company,
    TargetCompany,
    StrategicFit,
    Synergy,
    SynergyType,
    DealPriority,
)


logger = logging.getLogger(__name__)


@dataclass
class MarketCriteria:
    """Criteria for market scanning"""
    industries: List[str]
    size_range: tuple[float, float]  # revenue range in millions
    growth_rate_min: float = 0.0
    geographies: List[str] = None
    exclude_keywords: List[str] = None
    include_keywords: List[str] = None

    def __post_init__(self):
        if self.geographies is None:
            self.geographies = []
        if self.exclude_keywords is None:
            self.exclude_keywords = []
        if self.include_keywords is None:
            self.include_keywords = []


class DealSourcer:
    """
    Deal Sourcer Agent

    Continuously scans market for potential M&A targets:
    - Market scanning and monitoring
    - Company screening and filtering
    - Strategic fit analysis
    - Synergy identification
    - Target prioritization
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize Deal Sourcer

        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.scan_sources = self.config.get("scan_sources", [])
        self.market_criteria = self._load_criteria()

        # Knowledge base from past deals
        self.success_patterns = self.config.get("success_patterns", {})
        self.acquisition_history = self.config.get("acquisition_history", [])

        logger.info("Deal Sourcer initialized")

    def _load_criteria(self) -> List[MarketCriteria]:
        """Load market scanning criteria"""
        criteria_configs = self.config.get("market_criteria", [])

        if not criteria_configs:
            # Default criteria
            return [
                MarketCriteria(
                    industries=["Technology", "Software", "SaaS"],
                    size_range=(10, 500),  # $10M - $500M revenue
                    growth_rate_min=0.15,  # 15% minimum growth
                ),
            ]

        return [
            MarketCriteria(**criteria) for criteria in criteria_configs
        ]

    async def scan_market(
        self,
        force: bool = False,
    ) -> List[TargetCompany]:
        """
        Scan market for potential targets

        Args:
            force: Force a new scan even if recent scan exists

        Returns:
            List of potential target companies
        """
        logger.info("Scanning market for potential targets")

        targets = []

        # Scan across all market criteria
        for criteria in self.market_criteria:
            # In production, this would connect to data providers
            # like Crunchbase, PitchBook, Bloomberg, etc.
            companies = await self._scan_data_sources(criteria)

            # Screen companies against criteria
            screened = await self._screen_companies(companies, criteria)

            # Convert to target companies
            for company in screened:
                target = TargetCompany(
                    company_id=company.company_id,
                    name=company.name,
                    industry=company.industry,
                    sector=company.sector,
                    revenue=company.revenue,
                    ebitda=company.ebitda,
                    employees=company.employees,
                    growth_rate=company.growth_rate,
                    market_cap=company.market_cap,
                    description=company.description,
                    headquarters=company.headquarters,
                    website=company.website,
                    key_executives=company.key_executives,
                    products_services=company.products_services,
                    competitors=company.competitors,
                    financials=company.financials,
                    metadata=company.metadata,
                )

                targets.append(target)

        logger.info(f"Found {len(targets)} potential targets")

        return targets

    async def _scan_data_sources(
        self,
        criteria: MarketCriteria,
    ) -> List[Company]:
        """
        Scan data sources for companies matching criteria

        Args:
            criteria: Market criteria to match

        Returns:
            List of companies
        """
        # In production, integrate with:
        # - Crunchbase API
        # - PitchBook API
        # - Bloomberg Terminal
        # - Capital IQ
        # - FactSet
        # - PrivCo
        # - LinkedIn Sales Navigator

        # Mock implementation - in production replace with actual API calls
        companies = []

        # Simulate scanning delay
        await asyncio.sleep(0.1)

        # For now, return empty list
        # In production, this would query actual data sources
        return companies

    async def _screen_companies(
        self,
        companies: List[Company],
        criteria: MarketCriteria,
    ) -> List[Company]:
        """
        Screen companies against criteria

        Args:
            companies: Companies to screen
            criteria: Screening criteria

        Returns:
            Filtered list of companies
        """
        screened = []

        for company in companies:
            # Check revenue range
            if company.revenue:
                if not (criteria.size_range[0] <= company.revenue <= criteria.size_range[1]):
                    continue

            # Check growth rate
            if company.growth_rate and company.growth_rate < criteria.growth_rate_min:
                continue

            # Check industry
            if company.industry not in criteria.industries:
                continue

            # Check exclude keywords
            description = (company.description or "").lower()
            if any(kw.lower() in description for kw in criteria.exclude_keywords):
                continue

            # Check include keywords (if specified)
            if criteria.include_keywords:
                if not any(kw.lower() in description for kw in criteria.include_keywords):
                    continue

            screened.append(company)

        return screened

    async def analyze_strategic_fit(
        self,
        target: TargetCompany,
    ) -> StrategicFit:
        """
        Analyze strategic fit of target company

        Args:
            target: Target company to analyze

        Returns:
            StrategicFit analysis
        """
        logger.info(f"Analyzing strategic fit for {target.name}")

        # Analyze different fit dimensions
        strategic_alignment = await self._assess_strategic_alignment(target)
        cultural_fit = await self._assess_cultural_fit(target)
        technology_compatibility = await self._assess_technology_compatibility(target)
        market_expansion = await self._assess_market_expansion(target)

        # Identify synergies
        synergies = await self._identify_synergies(target)

        # Calculate overall score
        overall_score = (
            strategic_alignment * 0.3 +
            cultural_fit * 0.2 +
            technology_compatibility * 0.25 +
            market_expansion * 0.25
        )

        # Generate rationale
        rationale = self._generate_fit_rationale(
            target,
            strategic_alignment,
            cultural_fit,
            technology_compatibility,
            market_expansion,
            synergies,
        )

        # Identify concerns
        concerns = await self._identify_concerns(target)

        return StrategicFit(
            overall_score=overall_score,
            strategic_alignment=strategic_alignment,
            cultural_fit=cultural_fit,
            technology_compatibility=technology_compatibility,
            market_expansion=market_expansion,
            synergies=synergies,
            rationale=rationale,
            concerns=concerns,
        )

    async def _assess_strategic_alignment(self, target: TargetCompany) -> float:
        """Assess strategic alignment (0-1)"""
        # Check against strategic objectives
        score = 0.5  # Base score

        # Industry alignment
        if target.industry in self.config.get("target_industries", []):
            score += 0.2

        # Product/service complementarity
        if any(product in self.config.get("target_products", [])
               for product in target.products_services):
            score += 0.15

        # Market position
        if target.growth_rate and target.growth_rate > 0.2:
            score += 0.15

        return min(score, 1.0)

    async def _assess_cultural_fit(self, target: TargetCompany) -> float:
        """Assess cultural fit (0-1)"""
        # In production, use:
        # - Employee reviews (Glassdoor)
        # - Leadership style analysis
        # - Company values assessment
        # - Organizational structure

        score = 0.5  # Base score

        # Company size factor
        if target.employees:
            if 100 <= target.employees <= 1000:
                score += 0.2  # Good size for cultural integration

        # Headquarters location
        if target.headquarters:
            # Same geography often indicates better cultural fit
            score += 0.1

        return min(score, 1.0)

    async def _assess_technology_compatibility(self, target: TargetCompany) -> float:
        """Assess technology compatibility (0-1)"""
        score = 0.5  # Base score

        # Industry factor
        tech_industries = ["Technology", "Software", "SaaS", "Internet"]
        if any(ind in target.industry for ind in tech_industries):
            score += 0.3

        # Technology stack compatibility
        # In production, analyze actual tech stack
        if "cloud" in (target.description or "").lower():
            score += 0.1

        return min(score, 1.0)

    async def _assess_market_expansion(self, target: TargetCompany) -> float:
        """Assess market expansion potential (0-1)"""
        score = 0.5  # Base score

        # Geographic expansion
        if target.headquarters:
            # New market opportunity
            score += 0.2

        # Customer base expansion
        # In production, analyze customer overlap
        score += 0.15

        # Market share in sector
        if target.sector:
            score += 0.15

        return min(score, 1.0)

    async def _identify_synergies(
        self,
        target: TargetCompany,
    ) -> List[Synergy]:
        """Identify potential synergies"""
        synergies = []

        # Revenue synergies
        if target.industry in ["Technology", "Software"]:
            synergies.append(Synergy(
                synergy_type=SynergyType.REVENUE,
                description="Cross-selling opportunities to combined customer base",
                estimated_value=target.revenue * 0.1 if target.revenue else 0,
                confidence=0.6,
                time_to_realize=18,
            ))

        # Cost synergies
        if target.employees and target.employees > 100:
            synergies.append(Synergy(
                synergy_type=SynergyType.COST,
                description="G&A overhead reduction through shared services",
                estimated_value=target.ebitda * 0.15 if target.ebitda else 0,
                confidence=0.7,
                time_to_realize=12,
            ))

        # Technology synergies
        if "Technology" in target.industry:
            synergies.append(Synergy(
                synergy_type=SynergyType.TECHNOLOGY,
                description="Technology stack consolidation and modernization",
                estimated_value=target.revenue * 0.05 if target.revenue else 0,
                confidence=0.5,
                time_to_realize=24,
            ))

        # Talent synergies
        if target.key_executives:
            synergies.append(Synergy(
                synergy_type=SynergyType.TALENT,
                description="Key talent retention and knowledge transfer",
                estimated_value=target.revenue * 0.03 if target.revenue else 0,
                confidence=0.6,
                time_to_realize=6,
            ))

        return synergies

    def _generate_fit_rationale(
        self,
        target: TargetCompany,
        strategic_alignment: float,
        cultural_fit: float,
        technology_compatibility: float,
        market_expansion: float,
        synergies: List[Synergy],
    ) -> str:
        """Generate rationale for strategic fit"""
        rationale_parts = []

        # Strategic alignment
        if strategic_alignment > 0.7:
            rationale_parts.append(
                f"{target.name} shows strong strategic alignment "
                f"with our acquisition objectives in the {target.industry} sector."
            )

        # Cultural fit
        if cultural_fit > 0.7:
            rationale_parts.append(
                f"Cultural assessment indicates good fit for integration, "
                f"with compatible organizational values and structure."
            )

        # Technology
        if technology_compatibility > 0.7:
            rationale_parts.append(
                f"Technology stack appears compatible, enabling smooth integration "
                f"and rapid realization of technical synergies."
            )

        # Market expansion
        if market_expansion > 0.7:
            rationale_parts.append(
                f"Acquisition would provide significant market expansion opportunities "
                f"in the {target.sector} sector."
            )

        # Synergies
        total_synergy_value = sum(s.estimated_value for s in synergies)
        if total_synergy_value > 0:
            rationale_parts.append(
                f"Identified ${total_synergy_value:.1f}M in potential synergies "
                f"across {len(synergies)} categories."
            )

        return " ".join(rationale_parts) if rationale_parts else "Moderate strategic fit."

    async def _identify_concerns(self, target: TargetCompany) -> List[str]:
        """Identify potential concerns"""
        concerns = []

        # Integration complexity
        if target.employees and target.employees > 500:
            concerns.append(
                f"Large employee base ({target.employees}) "
                f"may increase integration complexity."
            )

        # Financial health
        if target.ebitda and target.revenue:
            margin = target.ebitda / target.revenue
            if margin < 0.1:
                concerns.append(
                    f"Low EBITDA margin ({margin:.1%}) indicates "
                    f"potential operational inefficiencies."
                )

        # Market concentration
        if len(target.competitors) < 3:
            concerns.append(
                "Highly concentrated market with few competitors "
                "may present antitrust considerations."
            )

        return concerns

    async def prioritize_targets(
        self,
        targets: List[TargetCompany],
        max_targets: int = 10,
    ) -> List[TargetCompany]:
        """
        Prioritize targets based on strategic fit and deal criteria

        Args:
            targets: List of target companies
            max_targets: Maximum number of targets to return

        Returns:
            Prioritized list of targets
        """
        # Score each target
        for target in targets:
            if not target.strategic_fit:
                target.strategic_fit = await self.analyze_strategic_fit(target)

            # Determine priority based on strategic fit
            if target.strategic_fit.overall_score >= 0.8:
                target.priority = DealPriority.CRITICAL
            elif target.strategic_fit.overall_score >= 0.7:
                target.priority = DealPriority.HIGH
            elif target.strategic_fit.overall_score >= 0.6:
                target.priority = DealPriority.MEDIUM
            else:
                target.priority = DealPriority.LOW

        # Sort by priority and strategic fit
        priority_order = {
            DealPriority.CRITICAL: 0,
            DealPriority.HIGH: 1,
            DealPriority.MEDIUM: 2,
            DealPriority.LOW: 3,
        }

        targets.sort(
            key=lambda t: (
                priority_order[t.priority],
                -t.strategic_fit.overall_score
            )
        )

        return targets[:max_targets]

    async def generate_acquisition_hypothesis(
        self,
        target: TargetCompany,
    ) -> str:
        """
        Generate acquisition hypothesis for target

        Args:
            target: Target company

        Returns:
            Acquisition hypothesis statement
        """
        hypothesis = f"""
        Acquisition Hypothesis for {target.name}

        Strategic Rationale:
        {target.strategic_fit.rationale if target.strategic_fit else 'Pending analysis'}

        Key Value Drivers:
        - Industry Position: {target.industry} / {target.sector}
        - Scale: ${target.revenue:.1f}M revenue, {target.employees} employees
        - Growth: {target.growth_rate:.1%} YoY growth rate

        Identified Synergies:
        """

        if target.strategic_fit and target.strategic_fit.synergies:
            for synergy in target.strategic_fit.synergies:
                hypothesis += f"\n- {synergy.synergy_type.value.capitalize()}: " \
                            f"${synergy.estimated_value:.1f}M " \
                            f"({synergy.time_to_realize} months)"

        return hypothesis
