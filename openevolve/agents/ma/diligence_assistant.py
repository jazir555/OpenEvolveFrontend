"""
M&A Diligence Assistant Agent

Automates due diligence process with checklist generation,
document analysis, risk identification, and reporting.
"""

import asyncio
import logging
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
from dataclasses import dataclass

from openevolve.agents.ma.schemas import (
    Deal,
    Company,
    DiligenceReport,
    DiligenceItem,
    DiligenceCategory,
    RiskFactor,
    RedFlag,
    RiskLevel,
)


logger = logging.getLogger(__name__)


@dataclass
class DiligencePlan:
    """Due diligence plan"""
    categories: List[DiligenceCategory]
    checklist: List[DiligenceItem]
    timeline: Dict[str, datetime]
    resource_requirements: Dict[str, int]
    estimated_duration: int  # days


class DiligenceAssistant:
    """
    Due Diligence Assistant

    Automates and manages the due diligence process:
    - Generates comprehensive diligence checklists
    - Analyzes documents and data
    - Identifies risks and red flags
    - Produces diligence reports
    - Tracks diligence progress
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize Diligence Assistant

        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.document_analyzers = self._init_analyzers()
        self.risk_patterns = self._load_risk_patterns()

        logger.info("Diligence Assistant initialized")

    def _init_analyzers(self) -> Dict[str, Any]:
        """Initialize document analyzers"""
        # In production, integrate with:
        # - Document AI (Google, Azure, AWS)
        # - Contract analysis platforms
        # - Financial analysis tools
        # - Background check services
        return {
            "financial": self._analyze_financial_documents,
            "legal": self._analyze_legal_documents,
            "contracts": self._analyze_contracts,
            "tax": self._analyze_tax_documents,
        }

    def _load_risk_patterns(self) -> Dict[str, List[Dict[str, Any]]]:
        """Load known risk patterns from past deals"""
        return self.config.get("risk_patterns", {})

    async def execute_diligence(
        self,
        deal: Deal,
        plan: Optional[Any] = None,
        depth: str = "comprehensive",
    ) -> DiligenceReport:
        """
        Execute due diligence for a deal

        Args:
            deal: Deal to conduct diligence on
            plan: Optional pre-evolved plan from LoongFlow
            depth: Depth of diligence (quick, standard, comprehensive)

        Returns:
            DiligenceReport: Comprehensive diligence report
        """
        logger.info(f"Executing {depth} diligence for deal {deal.deal_id}")

        # Generate checklist
        checklist = await self.generate_checklist(
            deal=deal,
            depth=depth,
        )

        # Simulate diligence execution
        # In production, this would actually collect and analyze documents
        await self._execute_checklist(checklist)

        # Analyze findings
        risks, red_flags = await self._identify_risks_and_flags(
            deal=deal,
            checklist=checklist,
        )

        # Identify opportunities
        opportunities = await self._identify_opportunities(deal, checklist)

        # Assess overall health
        financial_health = self._assess_financial_health(checklist)
        legal_compliance = self._assess_legal_compliance(checklist)
        operational_excellence = self._assess_operational_excellence(checklist)

        # Generate recommendation
        recommendation, confidence, rationale = self._generate_recommendation(
            risks=risks,
            red_flags=red_flags,
            opportunities=opportunities,
            financial_health=financial_health,
            legal_compliance=legal_compliance,
            operational_excellence=operational_excellence,
        )

        report = DiligenceReport(
            deal_id=deal.deal_id,
            target_company=deal.target_company.name,
            checklist=checklist,
            risks=risks,
            red_flags=red_flags,
            opportunities=opportunities,
            financial_health=financial_health,
            legal_compliance=legal_compliance,
            operational_excellence=operational_excellence,
            recommendation=recommendation,
            confidence=confidence,
            rationale=rationale,
        )

        logger.info(
            f"Diligence complete: {recommendation} "
            f"(confidence: {confidence:.1%})"
        )

        return report

    async def generate_checklist(
        self,
        deal: Deal,
        depth: str = "comprehensive",
    ) -> List[DiligenceItem]:
        """
        Generate due diligence checklist

        Args:
            deal: Deal context
            depth: Depth of diligence

        Returns:
            List of diligence checklist items
        """
        logger.info(f"Generating {depth} diligence checklist")

        checklist = []

        # Define items by category and depth
        items_by_depth = {
            "quick": self._quick_diligence_items(),
            "standard": self._standard_diligence_items(),
            "comprehensive": self._comprehensive_diligence_items(),
        }

        items = items_by_depth.get(depth, items_by_depth["standard"])

        # Create checklist items
        for category, category_items in items.items():
            for item_desc in category_items:
                item = DiligenceItem(
                    category=category,
                    item=item_desc,
                    status="pending",
                )
                checklist.append(item)

        # Customize based on deal characteristics
        checklist = await self._customize_checklist(
            deal=deal,
            checklist=checklist,
        )

        logger.info(f"Generated {len(checklist)} checklist items")

        return checklist

    def _quick_diligence_items(self) -> Dict[DiligenceCategory, List[str]]:
        """Quick diligence checklist items"""
        return {
            DiligenceCategory.FINANCIAL: [
                "Review last 3 years financial statements",
                "Analyze revenue trends",
                "Review major customers and concentration",
                "Assess working capital requirements",
            ],
            DiligenceCategory.LEGAL: [
                "Review material contracts",
                "Check for pending litigation",
                "Verify corporate structure and ownership",
                "Review IP ownership",
            ],
            DiligenceCategory.COMMERCIAL: [
                "Analyze market position",
                "Review customer base",
                "Assess competitive landscape",
            ],
            DiligenceCategory.OPERATIONAL: [
                "Review key operational processes",
                "Assess IT systems",
            ],
        }

    def _standard_diligence_items(self) -> Dict[DiligenceCategory, List[str]]:
        """Standard diligence checklist items"""
        return {
            DiligenceCategory.FINANCIAL: [
                "Review last 5 years financial statements",
                "Analyze revenue and margin trends",
                "Review customer concentration and contracts",
                "Assess working capital and cash flow",
                "Review debt and liabilities",
                "Analyze CAPEX requirements",
                "Review accounting policies and practices",
                "Assess financial projections",
            ],
            DiligenceCategory.LEGAL: [
                "Review all material contracts",
                "Check for pending and threatened litigation",
                "Verify corporate structure and ownership",
                "Review IP portfolio and ownership",
                "Check for regulatory compliance",
                "Review employment agreements",
                "Assess insurance coverage",
                "Review tax returns and assessments",
            ],
            DiligenceCategory.COMMERCIAL: [
                "Analyze market position and trends",
                "Review customer base and retention",
                "Assess competitive landscape",
                "Analyze product/service portfolio",
                "Review sales and marketing strategy",
                "Assess customer satisfaction",
            ],
            DiligenceCategory.OPERATIONAL: [
                "Review operational processes and KPIs",
                "Assess IT systems and infrastructure",
                "Review supply chain and vendors",
                "Assess manufacturing/service delivery",
                "Review facilities and equipment",
                "Assess operational risks",
            ],
            DiligenceCategory.TECHNOLOGY: [
                "Review technology stack and architecture",
                "Assess software quality and technical debt",
                "Review development processes",
                "Assess security practices",
                "Review IT team and capabilities",
            ],
            DiligenceCategory.HUMAN_RESOURCES: [
                "Review organizational structure",
                "Assess key talent and retention",
                "Review compensation and benefits",
                "Check for employment issues",
                "Assess culture and engagement",
            ],
            DiligenceCategory.TAX: [
                "Review tax returns (3 years)",
                "Assess tax positions and exposures",
                "Review tax attributes (NOLs, credits)",
                "Check for tax controversies",
                "Assess transfer pricing (if applicable)",
            ],
        }

    def _comprehensive_diligence_items(self) -> Dict[DiligenceCategory, List[str]]:
        """Comprehensive diligence checklist items"""
        return {
            DiligenceCategory.FINANCIAL: [
                "Review last 5 years financial statements and footnotes",
                "Analyze revenue trends by product/segment/geography",
                "Review customer contracts and concentration",
                "Assess working capital and cash flow patterns",
                "Review all debt instruments and covenants",
                "Analyze CAPEX history and requirements",
                "Review accounting policies and quality of earnings",
                "Assess financial projections and assumptions",
                "Review off-balance sheet arrangements",
                "Analyze working capital trends and normalization",
                "Review contingent liabilities and guarantees",
                "Assess pension and post-retirement obligations",
            ],
            DiligenceCategory.LEGAL: [
                "Review all material contracts and amendments",
                "Search for pending and threatened litigation",
                "Verify corporate structure and chain of title",
                "Review IP portfolio (patents, trademarks, copyrights)",
                "Check regulatory compliance and filings",
                "Review all employment and contractor agreements",
                "Assess insurance coverage and claims history",
                "Review tax returns and assessments (5 years)",
                "Check for environmental liabilities",
                "Review permits and licenses",
                "Assess real estate holdings and leases",
                "Review bankruptcy and insolvency history",
                "Check for FCPA or corruption issues",
            ],
            DiligenceCategory.COMMERCIAL: [
                "Analyze market size, growth, and trends",
                "Review customer base by segment and geography",
                "Assess competitive positioning and differentiation",
                "Analyze product/service portfolio and lifecycle",
                "Review sales strategy and funnel metrics",
                "Assess customer satisfaction and NPS",
                "Review pricing strategy and discounting",
                "Analyze market share and trends",
                "Review distribution channels and partners",
                "Assess brand strength and awareness",
            ],
            DiligenceCategory.OPERATIONAL: [
                "Review operational KPIs and benchmarks",
                "Assess IT systems and infrastructure",
                "Analyze supply chain and vendor relationships",
                "Review manufacturing/service delivery processes",
                "Assess facilities and equipment condition",
                "Review operational risk management",
                "Analyze inventory and logistics",
                "Review quality control processes",
                "Assess business continuity planning",
                "Review operational efficiency metrics",
            ],
            DiligenceCategory.TECHNOLOGY: [
                "Review complete technology stack",
                "Assess software architecture and quality",
                "Analyze technical debt and modernization needs",
                "Review development methodologies and practices",
                "Assess cybersecurity measures and incidents",
                "Review IT team skills and structure",
                "Analyze scalability and performance",
                "Review disaster recovery and backup",
                "Assess technology roadmap and investments",
                "Review open source and third-party dependencies",
                "Check for IP infringement risks",
            ],
            DiligenceCategory.HUMAN_RESOURCES: [
                "Review organizational structure and reporting",
                "Assess key talent and retention risk",
                "Analyze compensation and benefits programs",
                "Check for employment disputes and issues",
                "Assess employee engagement and culture",
                "Review union contracts (if applicable)",
                "Analyze turnover trends by department",
                "Review succession planning",
                "Assess diversity and inclusion metrics",
                "Review HR policies and procedures",
                "Check for misclassification issues",
            ],
            DiligenceCategory.TAX: [
                "Review tax returns and assessments (5 years)",
                "Analyze tax positions and uncertain tax benefits",
                "Review tax attributes (NOLs, credits, carryforwards)",
                "Check for ongoing tax controversies",
                "Assess transfer pricing documentation",
                "Review tax sharing agreements",
                "Analyze state and local tax exposures",
                "Review international tax structure",
                "Assess tax basis in assets",
                "Review sales tax and VAT compliance",
                "Check for unclaimed property issues",
            ],
        }

    async def _customize_checklist(
        self,
        deal: Deal,
        checklist: List[DiligenceItem],
    ) -> List[DiligenceItem]:
        """Customize checklist based on deal characteristics"""
        # Add industry-specific items
        if deal.target_company.industry == "Technology":
            checklist.extend([
                DiligenceItem(
                    category=DiligenceCategory.TECHNOLOGY,
                    item="Review software escrow arrangements",
                ),
                DiligenceItem(
                    category=DiligenceCategory.LEGAL,
                    item="Check for open source license compliance",
                ),
            ])

        # Add size-appropriate items
        if deal.target_company.employees and deal.target_company.employees > 500:
            checklist.append(
                DiligenceItem(
                    category=DiligenceCategory.HUMAN_RESOURCES,
                    item="Review ERISA and benefit plan compliance",
                )
            )

        # Add geography-specific items
        if deal.target_company.headquarters:
            if "Europe" in deal.target_company.headquarters:
                checklist.append(
                    DiligenceItem(
                        category=DiligenceCategory.LEGAL,
                        item="Review GDPR compliance",
                    )
                )

        return checklist

    async def _execute_checklist(
        self,
        checklist: List[DiligenceItem],
    ) -> None:
        """
        Execute checklist items

        In production, this would:
        - Request documents
        - Schedule expert reviews
        - Conduct interviews
        - Perform site visits
        - Run background checks
        """
        # Simulate execution
        for item in checklist:
            item.status = "complete"
            item.findings = f"Standard findings for {item.item}"

            # Simulate processing time
            await asyncio.sleep(0.001)

    async def _identify_risks_and_flags(
        self,
        deal: Deal,
        checklist: List[DiligenceItem],
    ) -> tuple[List[RiskFactor], List[RedFlag]]:
        """Identify risks and red flags"""
        risks = []
        red_flags = []

        # Analyze for risks
        risks.extend(await self._identify_financial_risks(deal, checklist))
        risks.extend(await self._identify_legal_risks(deal, checklist))
        risks.extend(await self._identify_operational_risks(deal, checklist))

        # Identify red flags
        red_flags.extend(await self._identify_red_flags(deal, checklist))

        return risks, red_flags

    async def _identify_financial_risks(
        self,
        deal: Deal,
        checklist: List[DiligenceItem],
    ) -> List[RiskFactor]:
        """Identify financial risks"""
        risks = []

        # Check for declining revenue
        if deal.target_company.growth_rate and deal.target_company.growth_rate < 0:
            risks.append(RiskFactor(
                category=DiligenceCategory.FINANCIAL,
                description="Declining revenue trend",
                level=RiskLevel.HIGH,
                mitigation="Review market conditions and competitive position",
            ))

        # Check for low margins
        if deal.target_company.ebitda and deal.target_company.revenue:
            margin = deal.target_company.ebitda / deal.target_company.revenue
            if margin < 0.1:
                risks.append(RiskFactor(
                    category=DiligenceCategory.FINANCIAL,
                    description=f"Low EBITDA margin ({margin:.1%})",
                    level=RiskLevel.MEDIUM,
                    mitigation="Focus on operational improvements and cost reduction",
                ))

        # Check for high customer concentration
        # In production, analyze actual customer data
        risks.append(RiskFactor(
            category=DiligenceCategory.FINANCIAL,
            description="Customer concentration risk (top 3 customers)",
            level=RiskLevel.MEDIUM,
            mitigation="Diversify customer base and secure key contracts",
        ))

        return risks

    async def _identify_legal_risks(
        self,
        deal: Deal,
        checklist: List[DiligenceItem],
    ) -> List[RiskFactor]:
        """Identify legal risks"""
        risks = []

        # Check for IP risks
        risks.append(RiskFactor(
            category=DiligenceCategory.LEGAL,
            description="Intellectual property protection and ownership",
            level=RiskLevel.MEDIUM,
            mitigation="Conduct thorough IP audit and secure assignments",
        ))

        # Check for litigation risk
        # In production, search actual litigation databases
        risks.append(RiskFactor(
            category=DiligenceCategory.LEGAL,
            description="Potential litigation exposure",
            level=RiskLevel.LOW,
            mitigation="Review insurance coverage and indemnifications",
        ))

        return risks

    async def _identify_operational_risks(
        self,
        deal: Deal,
        checklist: List[DiligenceItem],
    ) -> List[RiskFactor]:
        """Identify operational risks"""
        risks = []

        # Check for key person risk
        if deal.target_company.key_executives:
            risks.append(RiskFactor(
                category=DiligenceCategory.OPERATIONAL,
                description="Key person dependency in leadership",
                level=RiskLevel.MEDIUM,
                mitigation="Implement retention agreements and succession plan",
            ))

        # Check for IT risks
        risks.append(RiskFactor(
            category=DiligenceCategory.TECHNOLOGY,
            description="Technology integration complexity",
            level=RiskLevel.MEDIUM,
            mitigation="Detailed integration planning and technical assessment",
        ))

        return risks

    async def _identify_red_flags(
        self,
        deal: Deal,
        checklist: List[DiligenceItem],
    ) -> List[RedFlag]:
        """Identify critical red flags"""
        red_flags = []

        # Check for deal breakers
        # In production, analyze actual documents and data

        # Example red flags
        if deal.target_company.growth_rate and deal.target_company.growth_rate < -0.2:
            red_flags.append(RedFlag(
                category=DiligenceCategory.FINANCIAL,
                description="Revenue declining >20% YoY",
                severity=RiskLevel.CRITICAL,
                evidence=["Financial statement analysis"],
                recommendation="Carefully evaluate turnaround prospects",
                deal_breaker=True,
            ))

        return red_flags

    async def _identify_opportunities(
        self,
        deal: Deal,
        checklist: List[DiligenceItem],
    ) -> List[str]:
        """Identify value creation opportunities"""
        opportunities = []

        # Add opportunities based on analysis
        opportunities.append("Cost reduction through shared services")
        opportunities.append("Cross-selling to combined customer base")
        opportunities.append("Technology and process improvements")

        return opportunities

    def _assess_financial_health(self, checklist: List[DiligenceItem]) -> str:
        """Assess financial health"""
        return "Strong"

    def _assess_legal_compliance(self, checklist: List[DiligenceItem]) -> str:
        """Assess legal compliance"""
        return "Good"

    def _assess_operational_excellence(self, checklist: List[DiligenceItem]) -> str:
        """Assess operational excellence"""
        return "Adequate"

    def _generate_recommendation(
        self,
        risks: List[RiskFactor],
        red_flags: List[RedFlag],
        opportunities: List[str],
        financial_health: str,
        legal_compliance: str,
        operational_excellence: str,
    ) -> tuple[str, float, str]:
        """Generate diligence recommendation"""
        # Check for deal breakers
        deal_breakers = [rf for rf in red_flags if rf.deal_breaker]
        if deal_breakers:
            return (
                "reject",
                0.9,
                f"Deal breakers identified: {', '.join(rf.description for rf in deal_breakers)}"
            )

        # Count high/critical risks
        high_risks = [r for r in risks if r.level in [RiskLevel.HIGH, RiskLevel.CRITICAL]]
        if len(high_risks) > 5:
            return (
                "proceed_with_caution",
                0.5,
                f"Multiple high-risk items identified: {len(high_risks)}. "
                f"Require mitigation strategies before proceeding."
            )

        # Generate positive recommendation
        confidence = 0.8 - (len(high_risks) * 0.1)

        rationale = (
            f"Financial health: {financial_health}, "
            f"Legal compliance: {legal_compliance}, "
            f"Operational excellence: {operational_excellence}. "
            f"Identified {len(opportunities)} value creation opportunities."
        )

        return "proceed", max(confidence, 0.3), rationale

    # Document analysis methods (stubs for production implementation)
    async def _analyze_financial_documents(self, documents: List[str]) -> Dict[str, Any]:
        """Analyze financial documents"""
        return {}

    async def _analyze_legal_documents(self, documents: List[str]) -> Dict[str, Any]:
        """Analyze legal documents"""
        return {}

    async def _analyze_contracts(self, documents: List[str]) -> Dict[str, Any]:
        """Analyze contracts"""
        return {}

    async def _analyze_tax_documents(self, documents: List[str]) -> Dict[str, Any]:
        """Analyze tax documents"""
        return {}
