"""
M&A Deal Intelligence Platform - Data Schemas

Define canonical data models for the M&A deal workflow.
"""

from enum import Enum
from typing import Optional, List, Dict, Any
from datetime import datetime
from dataclasses import dataclass, field


class DealStage(str, Enum):
    """Stages in the M&A deal lifecycle"""
    SOURCING = "sourcing"
    DILIGENCE = "diligence"
    ANALYSIS = "analysis"
    DECISION = "decision"
    NEGOTIATION = "negotiation"
    CLOSING = "closing"
    INTEGRATION = "integration"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


class DealPriority(str, Enum):
    """Priority level for deals"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class SynergyType(str, Enum):
    """Types of synergies"""
    REVENUE = "revenue"
    COST = "cost"
    TECHNOLOGY = "technology"
    TALENT = "talent"
    MARKET = "market"


class RiskLevel(str, Enum):
    """Risk severity levels"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    NEGLIGIBLE = "negligible"


class DiligenceCategory(str, Enum):
    """Due diligence categories"""
    FINANCIAL = "financial"
    LEGAL = "legal"
    COMMERCIAL = "commercial"
    OPERATIONAL = "operational"
    TECHNOLOGY = "technology"
    HUMAN_RESOURCES = "human_resources"
    TAX = "tax"


@dataclass
class Company:
    """Company information"""
    company_id: str
    name: str
    industry: str
    sector: str
    revenue: Optional[float] = None
    ebitda: Optional[float] = None
    employees: Optional[int] = None
    growth_rate: Optional[float] = None
    market_cap: Optional[float] = None
    description: Optional[str] = None
    headquarters: Optional[str] = None
    website: Optional[str] = None
    key_executives: List[Dict[str, str]] = field(default_factory=list)
    products_services: List[str] = field(default_factory=list)
    competitors: List[str] = field(default_factory=list)
    financials: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Synergy:
    """Identified synergy"""
    synergy_type: SynergyType
    description: str
    estimated_value: float
    confidence: float
    time_to_realize: int  # months
    dependencies: List[str] = field(default_factory=list)
    risks: List[str] = field(default_factory=list)


@dataclass
class StrategicFit:
    """Strategic fit analysis"""
    overall_score: float
    strategic_alignment: float
    cultural_fit: float
    technology_compatibility: float
    market_expansion: float
    synergies: List[Synergy] = field(default_factory=list)
    rationale: str = ""
    concerns: List[str] = field(default_factory=list)


@dataclass
class TargetCompany(Company):
    """Target company for acquisition"""
    strategic_fit: Optional[StrategicFit] = None
    priority: DealPriority = DealPriority.MEDIUM
    acquisition_hypothesis: str = ""
    estimated_value_range: Optional[tuple[float, float]] = None
    contact_made: bool = False
    last_contact_date: Optional[datetime] = None
    notes: List[str] = field(default_factory=list)


@dataclass
class Deal:
    """M&A Deal"""
    deal_id: str
    target_company: TargetCompany
    stage: DealStage = DealStage.SOURCING
    priority: DealPriority = DealPriority.MEDIUM
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)

    # Deal team
    lead_advisor: Optional[str] = None
    team_members: List[str] = field(default_factory=list)

    # Deal parameters
    deal_size: Optional[float] = None
    currency: str = "USD"
    deal_type: Optional[str] = None  # asset_purchase, stock_purchase, merger

    # Status tracking
    stage_history: List[Dict[str, Any]] = field(default_factory=list)
    key_milestones: Dict[str, Optional[datetime]] = field(default_factory=dict)

    # Outcome
    outcome: Optional["DealOutcome"] = None

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RiskFactor:
    """Identified risk factor"""
    category: DiligenceCategory
    description: str
    level: RiskLevel
    mitigation: Optional[str] = None
    impact_description: Optional[str] = None


@dataclass
class RedFlag:
    """Critical red flag identified"""
    category: DiligenceCategory
    description: str
    severity: RiskLevel
    evidence: List[str] = field(default_factory=list)
    recommendation: str = ""
    deal_breaker: bool = False


@dataclass
class DiligenceItem:
    """Due diligence checklist item"""
    category: DiligenceCategory
    item: str
    status: str = "pending"  # pending, in_progress, complete, n/a
    assigned_to: Optional[str] = None
    due_date: Optional[datetime] = None
    findings: str = ""
    documents: List[str] = field(default_factory=list)
    risks: List[RiskFactor] = field(default_factory=list)


@dataclass
class DiligenceReport:
    """Due diligence report"""
    deal_id: str
    target_company: str
    generated_at: datetime = field(default_factory=datetime.utcnow)

    # Checklist
    checklist: List[DiligenceItem] = field(default_factory=list)

    # Findings
    risks: List[RiskFactor] = field(default_factory=list)
    red_flags: List[RedFlag] = field(default_factory=list)
    opportunities: List[str] = field(default_factory=list)

    # Assessment
    financial_health: Optional[str] = None
    legal_compliance: Optional[str] = None
    operational_excellence: Optional[str] = None

    # Recommendation
    recommendation: str = ""  # proceed, proceed_with_caution, reject
    confidence: float = 0.0
    rationale: str = ""

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ValuationMethod:
    """Valuation method and result"""
    method: str  # dcf, comps, precedent, asset
    value: float
    currency: str = "USD"
    assumptions: List[str] = field(default_factory=list)
    parameters: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0


@dataclass
class Scenario:
    """Valuation scenario"""
    name: str
    description: str
    probability: float
    valuation: float
    key_assumptions: Dict[str, Any] = field(default_factory=dict)
    sensitivities: Dict[str, float] = field(default_factory=dict)


@dataclass
class ValuationResult:
    """Valuation analysis result"""
    deal_id: str
    target_company: str
    valuation_date: datetime = field(default_factory=datetime.utcnow)

    # Valuation methods
    methods: List[ValuationMethod] = field(default_factory=list)
    implied_value: float = 0.0
    currency: str = "USD"

    # Scenarios
    scenarios: List[Scenario] = field(default_factory=list)
    base_case: float = 0.0
    best_case: float = 0.0
    worst_case: float = 0.0

    # Synergies
    identified_synergies: List[Synergy] = field(default_factory=list)
    synergy_value: float = 0.0

    # Analysis
    valuation_range: tuple[float, float] = (0.0, 0.0)
    confidence: float = 0.0
    key_value_drivers: List[str] = field(default_factory=list)
    risks: List[str] = field(default_factory=list)

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DealStructure:
    """Deal structure proposal"""
    deal_id: str
    structure_type: str  # stock, asset, merger

    # Consideration
    total_value: float
    currency: str = "USD"
    cash_component: float = 0.0
    stock_component: float = 0.0
    earnout: float = 0.0

    # Terms
    exchange_ratio: Optional[float] = None
    earnout_terms: Optional[Dict[str, Any]] = None
    payment_timeline: Optional[str] = None

    # Tax structure
    tax_structure: Optional[str] = None
    tax_benefits: List[str] = field(default_factory=list)

    # Risk allocation
    representations_warranties: List[str] = field(default_factory=list)
    indemnities: List[str] = field(default_factory=list)
    escrow: Optional[float] = None

    # Optimization metrics
    efficiency_score: float = 0.0
    risk_score: float = 0.0
    tax_efficiency: float = 0.0

    # Rationale
    rationale: str = ""
    alternatives: List[str] = field(default_factory=list)

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BATNA:
    """Best Alternative to Negotiated Agreement"""
    description: str
    value: float
    probability: float
    timeline: str = ""
    risks: List[str] = field(default_factory=list)


@dataclass
class NegotiationTactic:
    """Negotiation tactic recommendation"""
    tactic: str
    rationale: str
    timing: str
    expected_outcome: str
    risks: List[str] = field(default_factory=list)


@dataclass
class NegotiationStrategy:
    """Negotiation strategy and advice"""
    deal_id: str
    target_company: str
    created_at: datetime = field(default_factory=datetime.utcnow)

    # BATNA analysis
    batna: Optional[BATNA] = None
    their_batna: Optional[BATNA] = None

    # Strategy
    approach: str = ""  # collaborative, competitive, accommodating
    opening_position: Optional[Dict[str, Any]] = None
    target_position: Optional[Dict[str, Any]] = None
    fallback_position: Optional[Dict[str, Any]] = None

    # Tactics
    recommended_tactics: List[NegotiationTactic] = field(default_factory=list)

    # Key terms
    must_haves: List[str] = field(default_factory=list)
    nice_to_haves: List[str] = field(default_factory=list)
    tradeables: List[str] = field(default_factory=list)

    # Analysis
    leverage_assessment: str = ""
    game_theory_insights: List[str] = field(default_factory=list)

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class IntegrationMilestone:
    """Integration milestone"""
    name: str
    description: str
    target_date: datetime
    status: str = "pending"  # pending, in_progress, complete, delayed
    owner: Optional[str] = None
    dependencies: List[str] = field(default_factory=list)
    risks: List[str] = field(default_factory=list)
    success_criteria: List[str] = field(default_factory=list)


@dataclass
class SynergyRealization:
    """Synergy realization plan"""
    synergy: Synergy
    milestones: List[IntegrationMilestone] = field(default_factory=list)
    owner: Optional[str] = None
    tracking_metrics: List[str] = field(default_factory=list)
    risks: List[str] = field(default_factory=list)


@dataclass
class IntegrationPlan:
    """Post-merger integration plan"""
    deal_id: str
    target_company: str
    created_at: datetime = field(default_factory=datetime.utcnow)

    # Timeline
    day_1_plan: List[str] = field(default_factory=list)
    first_30_days: List[IntegrationMilestone] = field(default_factory=list)
    first_90_days: List[IntegrationMilestone] = field(default_factory=list)
    first_year: List[IntegrationMilestone] = field(default_factory=list)

    # Workstreams
    workstreams: Dict[str, List[IntegrationMilestone]] = field(default_factory=dict)

    # Synergy realization
    synergy_plans: List[SynergyRealization] = field(default_factory=list)

    # Risk management
    risks: List[RiskFactor] = field(default_factory=list)
    mitigation_strategies: List[str] = field(default_factory=list)

    # Change management
    communication_plan: List[str] = field(default_factory=list)
    retention_plan: List[str] = field(default_factory=list)
    culture_integration: List[str] = field(default_factory=list)

    # Governance
    steering_committee: List[str] = field(default_factory=list)
    reporting_structure: Optional[Dict[str, Any]] = None

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DealOutcome:
    """Final outcome of a deal"""
    deal_id: str
    outcome: str  # completed, withdrawn, rejected
    closed_date: Optional[datetime] = None

    # Financial outcomes
    final_value: Optional[float] = None
    currency: str = "USD"
    actual_vs_expected: float = 0.0  # percentage

    # Synergies
    synergies_realized: float = 0.0
    synergies_expected: float = 0.0
    synergy_realization_rate: float = 0.0

    # Integration
    integration_success: bool = False
    integration_timeline_adherence: float = 0.0

    # Lessons learned
    key_success_factors: List[str] = field(default_factory=list)
    key_challenges: List[str] = field(default_factory=list)
    lessons_learned: List[str] = field(default_factory=list)

    # Retrospective
    would_repeat: bool = False
    recommendation_for_future: str = ""

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
