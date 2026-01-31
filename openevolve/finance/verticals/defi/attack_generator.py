"""
DeFiAttackGenerator - Generate realistic DeFi attack scenarios

Generates comprehensive attack scenarios covering:
- Flash loan attacks
- Oracle manipulation
- Cascading liquidations
- Stablecoin de-pegs
- Smart contract bugs
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass

from openevolve.finance.verticals.defi.defi_evolver import DeFiAttackScenario


class DeFiAttackGenerator:
    """
    Generate realistic DeFi attack scenarios.

    Each scenario includes:
    - Attack steps
    - Expected profit
    - Attack vectors
    - Difficulty level
    """

    def generate_flash_loan_attack(self, assets: List[str]) -> DeFiAttackScenario:
        """
        Generate flash loan attack scenario.

        Classic attack:
        1. Borrow large amount via flash loan
        2. Use as collateral to borrow other asset
        3. Dump collateral asset on DEX
        4. Trigger liquidations
        5. Repay flash loan
        6. Keep profit
        """

        # Pick a volatile asset for the attack
        volatile_asset = "ETH" if "ETH" in assets else assets[0]
        stable_asset = "USDC" if "USDC" in assets else (assets[1] if len(assets) > 1 else assets[0])

        return DeFiAttackScenario(
            name="flash_loan_collateral_manipulation",
            description="Flash loan used to manipulate collateral price and trigger profitable liquidations",
            attack_type="flash_loan",
            attack_steps=[
                {
                    "step": 1,
                    "action": "flash_loan_borrow",
                    "asset": stable_asset,
                    "amount": 100_000_000,
                    "description": "Borrow $100M via flash loan"
                },
                {
                    "step": 2,
                    "action": "supply_collateral",
                    "asset": stable_asset,
                    "amount": 100_000_000,
                    "description": "Supply flash loan as collateral"
                },
                {
                    "step": 3,
                    "action": "borrow",
                    "asset": volatile_asset,
                    "amount": 50_000,
                    "collateral": stable_asset,
                    "description": "Borrow volatile asset at max CF"
                },
                {
                    "step": 4,
                    "action": "dump_on_dex",
                    "asset": volatile_asset,
                    "amount": 50_000,
                    "dex": "uniswap",
                    "price_impact": -0.30,
                    "description": "Dump asset on DEX, drop price 30%"
                },
                {
                    "step": 5,
                    "action": "trigger_liquidation",
                    "liquidate": stable_asset,
                    "receive": volatile_asset,
                    "description": "Trigger liquidations at distressed prices"
                },
                {
                    "step": 6,
                    "action": "repay_flash_loan",
                    "asset": stable_asset,
                    "amount": 100_000_000,
                    "description": "Repay flash loan"
                }
            ],
            expected_profit=5_000_000,
            attack_vectors=["oracle_manipulation", "liquidation_incentive", "flash_loan"],
            difficulty="hard"
        )

    def generate_oracle_manipulation(self, assets: List[str]) -> DeFiAttackScenario:
        """
        Generate oracle manipulation attack.

        Attacker manipulates spot price to exploit weak oracle.
        """

        target_asset = "ETH" if "ETH" in assets else assets[0]

        return DeFiAttackScenario(
            name="spot_price_oracle_pump",
            description="Manipulate spot price on CEX to exploit lending protocol oracle",
            attack_type="oracle_manipulation",
            attack_steps=[
                {
                    "step": 1,
                    "action": "wash_trading",
                    "exchange": "binance",
                    "pair": f"{target_asset}/USDT",
                    "volume": 1_000_000_000,
                    "price_increase": 0.50,
                    "description": "Wash trading to pump price 50%"
                },
                {
                    "step": 2,
                    "action": "supply_collateral",
                    "asset": target_asset,
                    "amount": 1000,
                    "description": "Supply pumped asset as collateral"
                },
                {
                    "step": 3,
                    "action": "borrow_max",
                    "asset": "USDC",
                    "collateral": target_asset,
                    "description": "Borrow max against inflated collateral"
                },
                {
                    "step": 4,
                    "action": "exit",
                    "keep": "USDC",
                    "description": "Exit position with profit"
                }
            ],
            expected_profit=2_000_000,
            attack_vectors=["oracle_manipulation", "wash_trading"],
            difficulty="medium"
        )

    def generate_cascading_liquidation(self, assets: List[str]) -> DeFiAttackScenario:
        """
        Generate cascading liquidation scenario.

        Simulates systemic risk event where one liquidation triggers others.
        """

        return DeFiAttackScenario(
            name="cascading_liquidation_cascade",
            description="Systemic price crash triggers cascading liquidations across protocol",
            attack_type="cascading_liquidation",
            attack_steps=[
                {
                    "step": 1,
                    "action": "market_crash",
                    "trigger": "external_shock",
                    "price_drop": 0.50,
                    "description": "External shock causes 50% price drop"
                },
                {
                    "step": 2,
                    "action": "mass_liquidation",
                    "affected_positions": "all_underwater",
                    "description": "All underwater positions get liquidated"
                },
                {
                    "step": 3,
                    "action": "sell_pressure",
                    "sell_volume": 100_000_000,
                    "description": "Liquidators dump collateral on market"
                },
                {
                    "step": 4,
                    "action": "further_price_drop",
                    "additional_drop": 0.20,
                    "description": "Selling pressure causes additional 20% drop"
                },
                {
                    "step": 5,
                    "action": "second_wave_liquidations",
                    "description": "Second wave of positions liquidated"
                }
            ],
            expected_profit=-50_000_000,  # Protocol loss, not attacker profit
            attack_vectors=["systemic_risk", "cascading_liquidation"],
            difficulty="extreme"
        )

    def generate_stablecoin_depeg(self, assets: List[str]) -> DeFiAttackScenario:
        """
        Generate stablecoin de-peg scenario.

        Simulates what happens when a major stablecoin loses its peg.
        """

        # Find stablecoins
        stablecoins = [asset for asset in assets if asset in ["USDC", "USDT", "DAI", "UST", "USDD"]]
        target_stable = stablecoins[0] if stablecoins else "USDC"

        return DeFiAttackScenario(
            name=f"{target_stable}_depeg_event",
            description=f"Major stablecoin {target_stable} loses its $1 peg",
            attack_type="stablecoin_depeg",
            attack_steps=[
                {
                    "step": 1,
                    "action": "depeg_trigger",
                    "stablecoin": target_stable,
                    "new_peg": 0.95,
                    "description": f"{target_stable} drops to $0.95"
                },
                {
                    "step": 2,
                    "action": "panic_redeem",
                    "volume": 500_000_000,
                    "description": "Users rush to redeem stablecoin"
                },
                {
                    "step": 3,
                    "action": "liquidity_crisis",
                    "description": "Protocol becomes illiquid"
                },
                {
                    "step": 4,
                    "action": "bad_debt_accumulation",
                    "description": "Bad debt accumulates as collateral value drops"
                }
            ],
            expected_profit=-100_000_000,
            attack_vectors=["stablecoin_risk", "liquidity_crisis"],
            difficulty="extreme"
        )

    def generate_reentrancy_attack(self, assets: List[str]) -> DeFiAttackScenario:
        """
        Generate reentrancy attack scenario.

        Classic smart contract vulnerability.
        """

        return DeFiAttackScenario(
            name="reentrancy_exploit",
            description="Reentrancy attack allows multiple withdrawals before balance update",
            attack_type="smart_contract_bug",
            attack_steps=[
                {
                    "step": 1,
                    "action": "deposit_collateral",
                    "asset": "ETH",
                    "amount": 1000,
                    "description": "Deposit collateral"
                },
                {
                    "step": 2,
                    "action": "borrow",
                    "asset": "USDC",
                    "amount": 500_000,
                    "description": "Borrow against collateral"
                },
                {
                    "step": 3,
                    "action": "malicious_contract_call",
                    "vulnerability": "reentrancy",
                    "description": "Call malicious contract during withdraw"
                },
                {
                    "step": 4,
                    "action": "recursive_withdraw",
                    "iterations": 10,
                    "description": "Recursively withdraw before balance update"
                },
                {
                    "step": 5,
                    "action": "drain_protocol",
                    "description": "Protocol funds drained"
                }
            ],
            expected_profit=10_000_000,
            attack_vectors=["smart_contract_bug", "reentrancy"],
            difficulty="hard"
        )

    def generate_historical_exploit_scenario(
        self,
        exploit_name: str,
        exploit_data: Dict[str, Any],
        assets: List[str]
    ) -> Optional[DeFiAttackScenario]:
        """
        Generate attack scenario based on historical exploit.

        Args:
            exploit_name: Name of historical exploit
            exploit_data: Data about the exploit
            assets: Available assets in protocol

        Returns:
            Attack scenario or None if cannot be generated
        """

        attack_type = exploit_data.get("attack_type", "unknown")

        if attack_type == "oracle_manipulation":
            return self._generate_oracle_exploit_scenario(exploit_name, exploit_data, assets)

        elif attack_type == "flash_loan":
            return self._generate_flash_loan_exploit_scenario(exploit_name, exploit_data, assets)

        elif attack_type == "smart_contract_bug":
            return self._generate_contract_bug_scenario(exploit_name, exploit_data, assets)

        elif attack_type == "liquidation":
            return self._generate_liquidation_exploit_scenario(exploit_name, exploit_data, assets)

        return None

    def _generate_oracle_exploit_scenario(
        self,
        name: str,
        data: Dict[str, Any],
        assets: List[str]
    ) -> DeFiAttackScenario:
        """Generate oracle manipulation exploit scenario"""

        description = data.get("description", "Historical oracle manipulation exploit")
        loss = data.get("loss_usd", 1_000_000)

        return DeFiAttackScenario(
            name=f"historical_{name}",
            description=f"Replay of {description}",
            attack_type="oracle_manipulation",
            attack_steps=[
                {
                    "step": 1,
                    "action": "identify_weak_oracle",
                    "oracle_type": "single_dex",
                    "description": "Identify protocol using single DEX as oracle"
                },
                {
                    "step": 2,
                    "action": "manipulate_price",
                    "method": "wash_trading",
                    "description": "Manipulate price on target DEX"
                },
                {
                    "step": 3,
                    "action": "exploit_protocol",
                    "description": "Borrow against inflated collateral value"
                },
                {
                    "step": 4,
                    "action": "exit_profit",
                    "description": "Exit with profit"
                }
            ],
            expected_profit=loss,
            attack_vectors=["oracle_manipulation", "historical_exploit"],
            difficulty="medium"
        )

    def _generate_flash_loan_exploit_scenario(
        self,
        name: str,
        data: Dict[str, Any],
        assets: List[str]
    ) -> DeFiAttackScenario:
        """Generate flash loan exploit scenario"""

        return self.generate_flash_loan_attack(assets)

    def _generate_contract_bug_scenario(
        self,
        name: str,
        data: Dict[str, Any],
        assets: List[str]
    ) -> DeFiAttackScenario:
        """Generate smart contract bug exploit scenario"""

        description = data.get("description", "Historical smart contract exploit")
        loss = data.get("loss_usd", 1_000_000)

        return DeFiAttackScenario(
            name=f"historical_{name}",
            description=f"Replay of {description}",
            attack_type="smart_contract_bug",
            attack_steps=[
                {
                    "step": 1,
                    "action": "identify_vulnerability",
                    "bug_type": "signature_bypass",
                    "description": "Identify signature verification bypass"
                },
                {
                    "step": 2,
                    "action": "craft_malicious_tx",
                    "description": "Craft transaction bypassing signature check"
                },
                {
                    "step": 3,
                    "action": "drain_funds",
                    "description": "Drain protocol funds"
                }
            ],
            expected_profit=loss,
            attack_vectors=["smart_contract_bug", "historical_exploit"],
            difficulty="hard"
        )

    def _generate_liquidation_exploit_scenario(
        self,
        name: str,
        data: Dict[str, Any],
        assets: List[str]
    ) -> DeFiAttackScenario:
        """Generate liquidation exploit scenario"""

        description = data.get("description", "Historical liquidation exploit")
        loss = data.get("loss_usd", 1_000_000)

        return DeFiAttackScenario(
            name=f"historical_{name}",
            description=f"Replay of {description}",
            attack_type="liquidation",
            attack_steps=[
                {
                    "step": 1,
                    "action": "identify_new_market",
                    "description": "Identify newly listed market with high CF"
                },
                {
                    "step": 2,
                    "action": "accumulate_tokens",
                    "description": "Accumulate large position in new market"
                },
                {
                    "step": 3,
                    "action": "trigger_liquidations",
                    "description": "Trigger massive liquidations due to low liquidity"
                },
                {
                    "step": 4,
                    "action": "buy_cheap_tokens",
                    "description": "Buy liquidated tokens below market value"
                }
            ],
            expected_profit=loss,
            attack_vectors=["liquidation", "market_manipulation", "historical_exploit"],
            difficulty="hard"
        )

    def generate_comprehensive_attack_suite(self, assets: List[str]) -> List[DeFiAttackScenario]:
        """
        Generate comprehensive suite of attack scenarios.

        Returns all major attack types for thorough testing.
        """

        scenarios = [
            self.generate_flash_loan_attack(assets),
            self.generate_oracle_manipulation(assets),
            self.generate_cascading_liquidation(assets),
            self.generate_stablecoin_depeg(assets),
            self.generate_reentrancy_attack(assets),
        ]

        # Add historical exploits
        for exploit_name, exploit_data in self._get_historical_exploits().items():
            scenario = self.generate_historical_exploit_scenario(
                exploit_name,
                exploit_data,
                assets
            )
            if scenario:
                scenarios.append(scenario)

        return scenarios

    def _get_historical_exploits(self) -> Dict[str, Dict[str, Any]]:
        """Get historical exploits database"""
        # This will be imported from historical_exploits.py
        # Placeholder for now
        return {}
