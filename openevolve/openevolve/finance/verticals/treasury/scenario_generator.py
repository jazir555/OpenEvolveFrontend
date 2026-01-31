"""
Liquidity Scenario Generator
Generate realistic liquidity stress scenarios for treasury testing

Author: AI Architecture Team
Date: 2026-01-30
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from enum import Enum
import numpy as np


class ScenarioType(Enum):
    """Types of liquidity scenarios"""
    GFC_CREDIT_FREEZE = "gfc_credit_freeze"
    SUPPLIER_CASCADE = "supplier_cascade"
    CP_MARKET_FREEZE = "cp_market_freeze"
    REVENUE_SHOCK = "revenue_shock"
    CAPEX_SURPRISE = "capex_surprise"
    COMBINED_STRESS = "combined_stress"


@dataclass
class LiquidityScenario:
    """
    Liquidity stress scenario

    Attributes:
        name: Scenario name
        description: Human-readable description
        scenario_type: Type of scenario
        duration_days: Duration in days
        daily_outflow: Daily outflow multiplier (1.0 = normal)
        cp_market_frozen: List of booleans for each day
        credit_line_frozen: List of booleans for each day
        supplier_payment_delays: List of days suppliers delay payments
        recovery_pattern: Recovery rate after crisis
    """
    name: str
    description: str
    scenario_type: ScenarioType
    duration_days: int
    daily_outflow: np.ndarray
    cp_market_frozen: List[bool]
    credit_line_frozen: List[bool]
    supplier_payment_delays: Optional[List[int]] = None
    recovery_pattern: str = "linear"  # linear, exponential, none

    def to_dict(self) -> Dict[str, Any]:
        """Convert scenario to dictionary"""
        return {
            'name': self.name,
            'description': self.description,
            'scenario_type': self.scenario_type.value,
            'duration_days': self.duration_days,
            'daily_outflow': self.daily_outflow.tolist(),
            'cp_market_frozen': self.cp_market_frozen,
            'credit_line_frozen': self.credit_line_frozen,
            'supplier_payment_delays': self.supplier_payment_delays,
            'recovery_pattern': self.recovery_pattern
        }


class LiquidityScenarioGenerator:
    """
    Generate realistic liquidity stress scenarios

    Based on historical treasury crises:
    - 2008 GFC: Credit freeze, CP market seizure
    - 2020 COVID: Revenue shock, supply chain disruption
    - Supplier defaults: Payment acceleration
    - Capex surprises: Urgent unplanned spend
    """

    def __init__(self, seed: Optional[int] = None):
        """
        Initialize scenario generator.

        Args:
            seed: Random seed for reproducibility
        """
        if seed is not None:
            np.random.seed(seed)

    def generate_gfc_credit_freeze(
        self,
        duration_days: int = 90
    ) -> LiquidityScenario:
        """
        Generate 2008 GFC credit freeze scenario.

        Key events:
        - Sept 2008: Lehman collapses
        - CP market freezes (no buyers)
        - Banks freeze credit lines
        - Suppliers demand cash payment (terms: net 30 -> cash in advance)

        Args:
            duration_days: Duration of crisis (default 90 days)

        Returns:
            LiquidityScenario representing GFC

        Example:
            >>> generator = LiquidityScenarioGenerator()
            >>> scenario = generator.generate_gfc_credit_freeze()
            >>> print(f"Duration: {scenario.duration_days} days")
            >>> print(f"Peak outflow: {max(scenario.daily_outflow):.1f}x normal")
        """

        # Day 14: Lehman collapses
        lehman_day = 14

        daily_outflow = np.ones(duration_days)
        cp_market_frozen = [False] * duration_days
        credit_line_frozen = [False] * duration_days

        for day in range(duration_days):
            if day < lehman_day:
                # Normal operations
                daily_outflow[day] = 1.0
            else:
                # Post-Lehman stress
                # Gradual increase to peak
                stress_factor = min(3.0, 1.0 + (day - lehman_day) * 0.1)
                daily_outflow[day] = stress_factor

                # CP market frozen after Lehman
                cp_market_frozen[day] = True

                # Credit lines frozen after Lehman
                credit_line_frozen[day] = True

        return LiquidityScenario(
            name="gfc_credit_freeze",
            description="2008 GFC: Lehman collapse triggers CP market freeze and credit line freeze",
            scenario_type=ScenarioType.GFC_CREDIT_FREEZE,
            duration_days=duration_days,
            daily_outflow=daily_outflow,
            cp_market_frozen=cp_market_frozen,
            credit_line_frozen=credit_line_frozen,
            supplier_payment_delays=None,
            recovery_pattern="exponential"
        )

    def generate_supplier_cascade(
        self,
        duration_days: int = 60
    ) -> LiquidityScenario:
        """
        Generate supplier default cascade scenario.

        Key events:
        - Major supplier defaults
        - Other suppliers demand advance payment (risk mitigation)
        - Sudden need to find new suppliers (prepayment required)
        - Gradual return to normal as new suppliers onboard

        Args:
            duration_days: Duration of crisis (default 60 days)

        Returns:
            LiquidityScenario representing supplier cascade

        Example:
            >>> scenario = generator.generate_supplier_cascade()
            >>> print(f"Peak outflow day: {np.argmax(scenario.daily_outflow)}")
        """

        daily_outflow = np.ones(duration_days)
        cp_market_frozen = [False] * duration_days
        credit_line_frozen = [False] * duration_days

        # Day 30: Major supplier defaults
        default_day = 30

        for day in range(duration_days):
            if day < default_day:
                # Normal operations
                daily_outflow[day] = 1.0
            elif day == default_day:
                # Sudden spike (find new suppliers, pay in advance)
                daily_outflow[day] = 5.0
            elif day < default_day + 14:
                # Gradual decrease as new suppliers onboard
                days_after_default = day - default_day
                daily_outflow[day] = 2.0 - (days_after_default * 0.05)
            else:
                # Return to normal
                daily_outflow[day] = 1.0

        return LiquidityScenario(
            name="supplier_cascade",
            description="Major supplier default triggers payment acceleration from other suppliers",
            scenario_type=ScenarioType.SUPPLIER_CASCADE,
            duration_days=duration_days,
            daily_outflow=daily_outflow,
            cp_market_frozen=cp_market_frozen,
            credit_line_frozen=credit_line_frozen,
            supplier_payment_delays=None,
            recovery_pattern="linear"
        )

    def generate_cp_market_freeze(
        self,
        duration_days: int = 45
    ) -> LiquidityScenario:
        """
        Generate CP market freeze scenario.

        Key events:
        - CP market seizes (no buyers for commercial paper)
        - Companies can't roll over CP maturing
        - Forced to use credit lines or cash
        - Credit lines still available (but at higher usage)

        Args:
            duration_days: Duration of freeze (default 45 days)

        Returns:
            LiquidityScenario representing CP market freeze

        Example:
            >>> scenario = generator.generate_cp_market_freeze()
            >>> print(f"CP frozen for {sum(scenario.cp_market_frozen)} days")
        """

        daily_outflow = np.ones(duration_days)
        cp_market_frozen = [False] * duration_days
        credit_line_frozen = [False] * duration_days

        # CP freeze starts at day 0, lasts 45 days
        for day in range(duration_days):
            # Normal operations
            daily_outflow[day] = 1.0

            # CP market frozen
            cp_market_frozen[day] = True

            # Credit lines still available
            credit_line_frozen[day] = False

        return LiquidityScenario(
            name="cp_market_freeze",
            description="Commercial paper market freezes, can't liquidate CP holdings",
            scenario_type=ScenarioType.CP_MARKET_FREEZE,
            duration_days=duration_days,
            daily_outflow=daily_outflow,
            cp_market_frozen=cp_market_frozen,
            credit_line_frozen=credit_line_frozen,
            supplier_payment_delays=None,
            recovery_pattern="exponential"
        )

    def generate_revenue_shock(
        self,
        duration_days: int = 90,
        shock_severity: float = 0.5
    ) -> LiquidityScenario:
        """
        Generate revenue shock scenario.

        Key events:
        - Sudden drop in revenue (e.g., COVID, economic downturn)
        - Fixed costs remain (burn rate increases relative to cash flow)
        - Gradual recovery as demand returns

        Args:
            duration_days: Duration of shock (default 90 days)
            shock_severity: Revenue drop as fraction (default 50%)

        Returns:
            LiquidityScenario representing revenue shock

        Example:
            >>> scenario = generator.generate_revenue_shock(shock_severity=0.6)
            >>> print(f"Worst outflow: {max(scenario.daily_outflow):.1f}x normal")
        """

        daily_outflow = np.ones(duration_days)
        cp_market_frozen = [False] * duration_days
        credit_line_frozen = [False] * duration_days

        # Shock starts at day 0
        shock_end_day = int(duration_days * 0.7)

        for day in range(duration_days):
            if day < shock_end_day:
                # During shock
                # Burn rate increases as revenue drops but costs stay fixed
                recovery_factor = day / shock_end_day
                effective_shock = shock_severity * (1 - recovery_factor * 0.5)
                daily_outflow[day] = 1.0 + effective_shock
            else:
                # Recovery phase
                recovery_progress = (day - shock_end_day) / (duration_days - shock_end_day)
                daily_outflow[day] = 1.0 + (shock_severity * 0.5 * (1 - recovery_progress))

        return LiquidityScenario(
            name="revenue_shock",
            description=f"Sudden revenue shock ({shock_severity*100:.0f}% drop) with gradual recovery",
            scenario_type=ScenarioType.REVENUE_SHOCK,
            duration_days=duration_days,
            daily_outflow=daily_outflow,
            cp_market_frozen=cp_market_frozen,
            credit_line_frozen=credit_line_frozen,
            supplier_payment_delays=None,
            recovery_pattern="linear"
        )

    def generate_capex_surprise(
        self,
        duration_days: int = 30
    ) -> LiquidityScenario:
        """
        Generate capex surprise scenario.

        Key events:
        - Urgent unplanned capital expenditure (e.g., equipment failure)
        - Large one-time outflow
        - Otherwise normal operations

        Args:
            duration_days: Duration (default 30 days)

        Returns:
            LiquidityScenario representing capex surprise

        Example:
            >>> scenario = generator.generate_capex_surprise()
            >>> print(f"Capex spike day: {np.argmax(scenario.daily_outflow)}")
        """

        daily_outflow = np.ones(duration_days)
        cp_market_frozen = [False] * duration_days
        credit_line_frozen = [False] * duration_days

        # Capex occurs on day 7
        capex_day = 7
        capex_multiplier = 10.0  # 10x normal daily burn

        for day in range(duration_days):
            if day == capex_day:
                daily_outflow[day] = capex_multiplier
            else:
                daily_outflow[day] = 1.0

        return LiquidityScenario(
            name="capex_surprise",
            description="Urgent unplanned capital expenditure (equipment failure, compliance)",
            scenario_type=ScenarioType.CAPEX_SURPRISE,
            duration_days=duration_days,
            daily_outflow=daily_outflow,
            cp_market_frozen=cp_market_frozen,
            credit_line_frozen=credit_line_frozen,
            supplier_payment_delays=None,
            recovery_pattern="none"
        )

    def generate_combined_stress(
        self,
        duration_days: int = 120
    ) -> LiquidityScenario:
        """
        Generate combined stress scenario (worst case).

        Key events:
        - Revenue shock
        - CP market freeze
        - Supplier payment acceleration
        - Credit line partially frozen

        Args:
            duration_days: Duration (default 120 days)

        Returns:
            LiquidityScenario representing combined stress

        Example:
            >>> scenario = generator.generate_combined_stress()
            >>> print(f"Peak outflow: {max(scenario.daily_outflow):.1f}x normal")
        """

        daily_outflow = np.ones(duration_days)
        cp_market_frozen = [False] * duration_days
        credit_line_frozen = [False] * duration_days

        # Crisis starts at day 30
        crisis_start = 30

        for day in range(duration_days):
            if day < crisis_start:
                # Normal operations
                daily_outflow[day] = 1.0
            else:
                days_crisis = day - crisis_start

                # Combined stress increases over time
                stress_factor = 1.0 + min(2.5, days_crisis * 0.05)
                daily_outflow[day] = stress_factor

                # CP market frozen after crisis starts
                cp_market_frozen[day] = True

                # Credit lines 50% frozen (banks conserving capital)
                credit_line_frozen[day] = (np.random.random() < 0.5)

        return LiquidityScenario(
            name="combined_stress",
            description="Combined stress: Revenue shock + CP freeze + partial credit line freeze",
            scenario_type=ScenarioType.COMBINED_STRESS,
            duration_days=duration_days,
            daily_outflow=daily_outflow,
            cp_market_frozen=cp_market_frozen,
            credit_line_frozen=credit_line_frozen,
            supplier_payment_delays=None,
            recovery_pattern="exponential"
        )

    def generate_all_scenarios(self) -> List[LiquidityScenario]:
        """
        Generate all standard scenarios.

        Returns:
            List of all liquidity scenarios

        Example:
            >>> scenarios = generator.generate_all_scenarios()
            >>> print(f"Generated {len(scenarios)} scenarios")
            >>> for scenario in scenarios:
            ...     print(f"  - {scenario.name}")
        """

        return [
            self.generate_gfc_credit_freeze(),
            self.generate_supplier_cascade(),
            self.generate_cp_market_freeze(),
            self.generate_revenue_shock(),
            self.generate_capex_surprise(),
            self.generate_combined_stress()
        ]

    def generate_custom_scenario(
        self,
        name: str,
        description: str,
        duration_days: int,
        outflow_pattern: str = "constant",
        outflow_parameters: Optional[Dict[str, float]] = None,
        cp_freeze_start: Optional[int] = None,
        cp_freeze_duration: Optional[int] = None,
        credit_freeze_start: Optional[int] = None,
        credit_freeze_duration: Optional[int] = None
    ) -> LiquidityScenario:
        """
        Generate custom liquidity scenario.

        Args:
            name: Scenario name
            description: Scenario description
            duration_days: Duration in days
            outflow_pattern: Pattern type ('constant', 'spike', 'gradual_increase', 'oscillating')
            outflow_parameters: Parameters for outflow pattern
            cp_freeze_start: Day when CP market freezes (None = no freeze)
            cp_freeze_duration: How long CP market stays frozen
            credit_freeze_start: Day when credit lines freeze (None = no freeze)
            credit_freeze_duration: How long credit lines stay frozen

        Returns:
            Custom LiquidityScenario

        Example:
            >>> scenario = generator.generate_custom_scenario(
            ...     name="custom_stress",
            ...     description="Custom stress scenario",
            ...     duration_days=60,
            ...     outflow_pattern="gradual_increase",
            ...     outflow_parameters={'start': 1.0, 'peak': 2.5, 'peak_day': 30},
            ...     cp_freeze_start=10,
            ...     cp_freeze_duration=30
            ... )
        """

        if outflow_parameters is None:
            outflow_parameters = {}

        daily_outflow = np.ones(duration_days)
        cp_market_frozen = [False] * duration_days
        credit_line_frozen = [False] * duration_days

        # Generate outflow pattern
        if outflow_pattern == "constant":
            multiplier = outflow_parameters.get('multiplier', 1.0)
            daily_outflow[:] = multiplier

        elif outflow_pattern == "spike":
            spike_day = int(outflow_parameters.get('spike_day', duration_days // 2))
            spike_multiplier = outflow_parameters.get('spike_multiplier', 5.0)
            daily_outflow[spike_day] = spike_multiplier

        elif outflow_pattern == "gradual_increase":
            start = outflow_parameters.get('start', 1.0)
            peak = outflow_parameters.get('peak', 2.5)
            peak_day = int(outflow_parameters.get('peak_day', duration_days // 2))

            for day in range(duration_days):
                if day < peak_day:
                    progress = day / peak_day
                    daily_outflow[day] = start + (peak - start) * progress
                else:
                    daily_outflow[day] = peak

        elif outflow_pattern == "oscillating":
            amplitude = outflow_parameters.get('amplitude', 0.5)
            frequency = outflow_parameters.get('frequency', 0.1)
            base = outflow_parameters.get('base', 1.5)

            for day in range(duration_days):
                daily_outflow[day] = base + amplitude * np.sin(2 * np.pi * frequency * day)

        # Apply CP freeze
        if cp_freeze_start is not None and cp_freeze_duration is not None:
            end = min(cp_freeze_start + cp_freeze_duration, duration_days)
            for day in range(cp_freeze_start, end):
                cp_market_frozen[day] = True

        # Apply credit line freeze
        if credit_freeze_start is not None and credit_freeze_duration is not None:
            end = min(credit_freeze_start + credit_freeze_duration, duration_days)
            for day in range(credit_freeze_start, end):
                credit_line_frozen[day] = True

        return LiquidityScenario(
            name=name,
            description=description,
            scenario_type=ScenarioType.COMBINED_STRESS,
            duration_days=duration_days,
            daily_outflow=daily_outflow,
            cp_market_frozen=cp_market_frozen,
            credit_line_frozen=credit_line_frozen,
            supplier_payment_delays=None,
            recovery_pattern="linear"
        )
