"""
DeFiProtocolSimulator - Simulate lending protocol behavior and attacks
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
import numpy as np

from openevolve.finance.verticals.defi.defi_evolver import (
    ProtocolParameters,
    DeFiAttackScenario,
    DeFiAttackResult,
    HistoricalSimulation,
)


@dataclass
class ProtocolState:
    """State of lending protocol during simulation"""
    total_supplied: float = 0.0
    total_borrows: float = 0.0
    total_collateral: float = 0.0
    bad_debt: float = 0.0
    utilization: float = 0.0
    prices: Dict[str, float] = field(default_factory=dict)
    user_positions: List[Dict[str, Any]] = field(default_factory=list)
    reserve_factors: Dict[str, float] = field(default_factory=dict)


@dataclass
class UserPosition:
    """User position in protocol"""
    user_id: str
    supplied: Dict[str, float]  # asset -> amount
    borrowed: Dict[str, float]  # asset -> amount
    collateral_factor: float
    health_factor: float = 1.0


class DeFiProtocolSimulator:
    """
    Simulate DeFi lending protocol operations and attacks.

    This simulator models:
    - Supply/borrow mechanics
    - Collateralization and liquidation
    - Price oracle behavior
    - Flash loan execution
    - Attack vectors
    """

    def __init__(self):
        self.tvl = 1_000_000_000  # $1B TVL baseline
        self.user_count = 10000
        self.base_prices = {
            "ETH": 3000.0,
            "USDC": 1.0,
            "WBTC": 50000.0,
            "USDT": 1.0,
            "DAI": 1.0,
        }

    async def simulate_attack(
        self,
        parameters: ProtocolParameters,
        protocol: str,
        assets: List[str],
        attack: DeFiAttackScenario
    ) -> DeFiAttackResult:
        """
        Simulate attack on protocol with given parameters.

        Returns:
            Attack result with:
            - survived: Did protocol survive?
            - attacker_profit: Profit made by attacker
            - protocol_loss: Funds lost by protocol
            - bad_debt: Remaining bad debt
        """
        # Initialize protocol state
        state = self._initialize_protocol(
            parameters=parameters,
            protocol=protocol,
            assets=assets
        )

        initial_bad_debt = state.bad_debt

        # Execute attack steps
        for step in attack.attack_steps:
            state = await self._execute_attack_step(
                state=state,
                step=step,
                parameters=parameters
            )

            # Check if protocol already failed
            if state.bad_debt > parameters.min_liquidity_required * 0.1:  # 10% of min liquidity
                return DeFiAttackResult(
                    survived=False,
                    attacker_profit=attack.expected_profit,
                    protocol_loss=state.bad_debt - initial_bad_debt,
                    bad_debt=state.bad_debt,
                    failure_point=step,
                    capital_efficiency=state.utilization,
                    utilization=state.utilization
                )

        # Protocol survived
        return DeFiAttackResult(
            survived=True,
            attacker_profit=0,  # Attack failed
            protocol_loss=0,
            bad_debt=state.bad_debt,
            failure_point=None,
            capital_efficiency=state.utilization,
            utilization=state.utilization
        )

    async def simulate_history(
        self,
        parameters: ProtocolParameters,
        protocol: str,
        assets: List[str]
    ) -> HistoricalSimulation:
        """Simulate protocol performance through historical events"""

        # Historical events to simulate
        events = [
            {
                "date": "2020-03-12",
                "name": "covid_crash",
                "description": "COVID market crash",
                "price_changes": {"ETH": -0.50, "WBTC": -0.45},
                "volatility": 0.30
            },
            {
                "date": "2021-05-19",
                "name": "china_mining_ban",
                "description": "China mining ban crash",
                "price_changes": {"ETH": -0.40, "WBTC": -0.35},
                "volatility": 0.25
            },
            {
                "date": "2022-05-09",
                "name": "ust_depeg",
                "description": "UST de-peg event",
                "price_changes": {},
                "stablecoin_depeg": "UST",
                "depeg_amount": -0.95,
                "volatility": 0.40
            },
            {
                "date": "2022-11-11",
                "name": "ftx_collapse",
                "description": "FTX collapse",
                "price_changes": {"ETH": -0.20, "WBTC": -0.15},
                "volatility": 0.20
            },
            {
                "date": "2023-03-11",
                "name": "svb_crisis",
                "description": "SVB banking crisis",
                "price_changes": {},
                "stablecoin_depeg": "USDC",
                "depeg_amount": -0.10,
                "volatility": 0.15
            },
        ]

        results = []

        for event in events:
            result = await self._simulate_event(
                parameters=parameters,
                protocol=protocol,
                assets=assets,
                event=event
            )
            results.append(result)

        # Calculate aggregate metrics
        avg_utilization = np.mean([r["utilization"] for r in results])
        max_bad_debt = max([r["bad_debt"] for r in results])
        survived_all = all([r["survived"] for r in results])

        return HistoricalSimulation(
            event_results=results,
            avg_utilization=avg_utilization,
            max_bad_debt=max_bad_debt,
            survived_all_events=survived_all,
            total_events=len(events)
        )

    def _initialize_protocol(
        self,
        parameters: ProtocolParameters,
        protocol: str,
        assets: List[str]
    ) -> ProtocolState:
        """Initialize protocol state with realistic values"""

        # Initialize prices
        prices = {asset: self.base_prices.get(asset, 1.0) for asset in assets}

        # Generate user positions
        user_positions = []
        for i in range(self.user_count):
            # Random supply amounts
            supplied = {
                asset: np.random.uniform(0, 100000)
                for asset in assets
            }

            # Calculate borrow power based on collateral factors
            total_collateral_value = sum(
                supplied[asset] * prices[asset] * parameters.collateral_factors.get(asset, 0.75)
                for asset in assets
            )

            # Borrow up to 80% of borrow power on average
            borrowed = {}
            if total_collateral_value > 0:
                for asset in assets:
                    max_borrow = total_collateral_value / len(assets)
                    borrowed[asset] = np.random.uniform(0, max_borrow / prices[asset])

            user_positions.append({
                "user_id": f"user_{i}",
                "supplied": supplied,
                "borrowed": borrowed,
                "health_factor": np.random.uniform(1.5, 5.0)
            })

        # Calculate totals
        total_supplied = sum(
            sum(pos["supplied"].values()) for pos in user_positions
        )
        total_borrows = sum(
            sum(pos["borrowed"].values()) for pos in user_positions
        )

        utilization = total_borrows / total_supplied if total_supplied > 0 else 0

        return ProtocolState(
            total_supplied=total_supplied,
            total_borrows=total_borrows,
            total_collateral=total_supplied - total_borrows,
            bad_debt=0.0,
            utilization=utilization,
            prices=prices,
            user_positions=user_positions,
            reserve_factors={asset: 0.1 for asset in assets}  # 10% reserve factor
        )

    async def _execute_attack_step(
        self,
        state: ProtocolState,
        step: Dict[str, Any],
        parameters: ProtocolParameters
    ) -> ProtocolState:
        """Execute one step of an attack"""

        action = step.get("action")

        if action == "flash_loan_borrow":
            # Attacker borrows large amount
            state = self._handle_flash_loan(state, step)

        elif action == "supply_collateral":
            # Attacker supplies collateral
            state = self._handle_supply_collateral(state, step)

        elif action == "borrow":
            # Attacker borrows against collateral
            state = self._handle_borrow(state, step, parameters)

        elif action == "dump_on_dex":
            # Attacker dumps asset to manipulate price
            state = self._handle_price_manipulation(state, step, parameters)

        elif action == "trigger_liquidation":
            # Trigger liquidations
            state = self._handle_liquidation(state, step, parameters)

        elif action == "wash_trading":
            # Wash trading to manipulate price
            state = self._handle_wash_trading(state, step, parameters)

        elif action == "borrow_max":
            # Borrow maximum possible
            state = self._handle_borrow_max(state, step, parameters)

        return state

    def _handle_flash_loan(self, state: ProtocolState, step: Dict[str, Any]) -> ProtocolState:
        """Handle flash loan borrow"""
        # Flash loans don't permanently change state if repaid
        # Track for attack validation
        return state

    def _handle_supply_collateral(self, state: ProtocolState, step: Dict[str, Any]) -> ProtocolState:
        """Handle supply collateral"""
        asset = step.get("asset")
        amount = step.get("amount", 0)

        state.total_supplied += amount
        state.total_collateral += amount

        return state

    def _handle_borrow(
        self,
        state: ProtocolState,
        step: Dict[str, Any],
        parameters: ProtocolParameters
    ) -> ProtocolState:
        """Handle borrow against collateral"""
        asset = step.get("asset")
        amount = step.get("amount", 0)
        collateral = step.get("collateral")

        # Calculate borrow power
        collateral_value = amount * state.prices.get(collateral, 1.0)
        cf = parameters.collateral_factors.get(collateral, 0.75)
        borrow_power = collateral_value * cf

        # Check if borrow is valid
        borrow_value = amount * state.prices.get(asset, 1.0)

        if borrow_value <= borrow_power:
            state.total_borrows += borrow_value

            # Check if undercollateralized (bad debt)
            if borrow_value > borrow_power * 0.9:  # Too close to limit
                state.bad_debt += max(0, borrow_value - borrow_power)

        return state

    def _handle_price_manipulation(
        self,
        state: ProtocolState,
        step: Dict[str, Any],
        parameters: ProtocolParameters
    ) -> ProtocolState:
        """Handle price manipulation via DEX dump"""
        asset = step.get("asset")
        price_impact = step.get("price_impact", -0.30)

        # Check circuit breaker
        if abs(price_impact) > parameters.circuit_breaker_threshold:
            # Circuit breaker triggered - reject trade
            return state

        # Update price
        old_price = state.prices.get(asset, 1.0)
        new_price = old_price * (1 + price_impact)
        state.prices[asset] = new_price

        # Check if this triggers liquidations
        state = self._check_liquidations(state, parameters)

        return state

    def _handle_wash_trading(
        self,
        state: ProtocolState,
        step: Dict[str, Any],
        parameters: ProtocolParameters
    ) -> ProtocolState:
        """Handle wash trading manipulation"""
        asset = step.get("asset")
        price_increase = step.get("price_increase", 0.50)

        # TWAP oracles mitigate wash trading
        if parameters.price_oracle_type in ["twap", "median", "chainlink"]:
            # Reduced impact
            price_increase *= 0.1

        # Spot oracles are vulnerable
        if parameters.price_oracle_type == "spot":
            # Check circuit breaker
            if abs(price_increase) > parameters.circuit_breaker_threshold:
                return state  # Circuit breaker

            # Update price
            old_price = state.prices.get(asset, 1.0)
            state.prices[asset] = old_price * (1 + price_increase)

        return state

    def _handle_borrow_max(
        self,
        state: ProtocolState,
        step: Dict[str, Any],
        parameters: ProtocolParameters
    ) -> ProtocolState:
        """Handle maximum borrow"""
        asset = step.get("asset")
        collateral = step.get("collateral")

        # Calculate max borrow
        collateral_value = 1000000  # Assume $1M collateral
        cf = parameters.collateral_factors.get(collateral, 0.75)
        max_borrow = collateral_value * cf

        state.total_borrows += max_borrow

        # High risk of bad debt
        if cf > 0.80:
            state.bad_debt += max_borrow * 0.1

        return state

    def _handle_liquidation(
        self,
        state: ProtocolState,
        step: Dict[str, Any],
        parameters: ProtocolParameters
    ) -> ProtocolState:
        """Handle liquidation event"""
        # Liquidations reduce bad debt if done correctly
        # But cascading liquidations can increase bad debt

        # Simulate liquidation efficiency
        liquidation_bonus = parameters.liquidation_bonuses.get(
            step.get("liquidate", ""),
            0.08
        )

        # Higher bonus = more liquidators = less bad debt
        if liquidation_bonus > 0.10:
            # Good liquidation
            state.bad_debt *= 0.5
        else:
            # Poor liquidation - cascading
            state.bad_debt *= 1.5

        return state

    def _check_liquidations(
        self,
        state: ProtocolState,
        parameters: ProtocolParameters
    ) -> ProtocolState:
        """Check and execute liquidations"""
        for position in state.user_positions:
            # Calculate health factor
            collateral_value = sum(
                position["supplied"].get(asset, 0) * state.prices.get(asset, 1.0)
                for asset in position["supplied"]
            )

            borrow_value = sum(
                position["borrowed"].get(asset, 0) * state.prices.get(asset, 1.0)
                for asset in position["borrowed"]
            )

            if collateral_value > 0:
                health_factor = collateral_value / borrow_value if borrow_value > 0 else 10.0

                # Liquidate if health factor < 1.0
                if health_factor < 1.0:
                    # Liquidation happens
                    liquidation_bonus = 0.08  # Average bonus

                    # Calculate bad debt if collateral < borrow
                    if collateral_value < borrow_value:
                        state.bad_debt += borrow_value - collateral_value

                    # Reduce borrows
                    for asset in position["borrowed"]:
                        position["borrowed"][asset] *= 0.5

                    position["health_factor"] = health_factor

        return state

    async def _simulate_event(
        self,
        parameters: ProtocolParameters,
        protocol: str,
        assets: List[str],
        event: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Simulate one historical event"""

        # Initialize state
        state = self._initialize_protocol(parameters, protocol, assets)

        # Apply event effects
        price_changes = event.get("price_changes", {})

        # Apply price changes
        for asset, change in price_changes.items():
            if asset in state.prices:
                state.prices[asset] *= (1 + change)

        # Handle stablecoin de-peg
        if "stablecoin_depeg" in event:
            sc = event["stablecoin_depeg"]
            depeg = event.get("depeg_amount", -0.50)
            if sc in state.prices:
                state.prices[sc] = max(0.01, state.prices[sc] * (1 + depeg))

        # Check liquidations
        state = self._check_liquidations(state, parameters)

        # Calculate results
        survived = state.bad_debt < parameters.min_liquidity_required * 0.05  # 5% threshold

        return {
            "event": event["name"],
            "date": event["date"],
            "survived": survived,
            "bad_debt": state.bad_debt,
            "utilization": state.utilization,
            "price_impact": price_changes,
        }
