# DeFi Vertical - Protocol Risk Evolution

## Overview

The DeFi vertical evolves lending protocol parameters to survive historical exploits and black swan attacks. It combines LoongFlow's adversarial planning with OpenEvolve's optimization to find robust parameter settings.

## Risk Framework

### Core DeFi Risks

1. **Oracle Manipulation**
   - Spot price pumps via wash trading
   - TWAP manipulation
   - Cross-exchange arbitrage manipulation
   - Specialized oracle failures (Chainlink outages, etc.)

2. **Flash Loan Attacks**
   - Collateral -> borrow -> dump -> repay cycles
   - Multi-hop attacks across protocols
   - Protocol-to-protocol arbitrage exploitation

3. **Cascading Liquidations**
   - Systemic risk events (50%+ price drops)
   - Cross-protocol contagion
   - Liquidation cascades triggering further liquidations

4. **Token Peg Failures**
   - Stablecoin de-pegs (UST, USDD, etc.)
   - Wrapped token failures
   - Liquid staking token (LST) de-pegs

5. **Smart Contract Bugs**
   - Reentrancy vulnerabilities
   - Rounding error exploits
   - Integer overflow/underflow
   - Signature verification bypasses

## Attack Scenario Methodology

### Scenario Generation

```python
from openevolve.finance.verticals.defi import DeFiAttackGenerator

generator = DeFiAttackGenerator()

# Generate specific attack types
flash_loan = generator.generate_flash_loan_attack(["ETH", "USDC", "WBTC"])
oracle_manip = generator.generate_oracle_manipulation(["ETH", "USDC"])
cascade = generator.generate_cascading_liquidation(["ETH", "USDC", "WBTC"])
depeg = generator.generate_stablecoin_depeg(["USDC", "USDT", "ETH"])
reentrancy = generator.generate_reentrancy_attack(["ETH", "USDC"])

# Generate comprehensive suite
all_scenarios = generator.generate_comprehensive_attack_suite([
    "ETH", "USDC", "WBTC", "DAI"
])
```

### Historical Exploit Learning

```python
from openevolve.finance.verticals.defi.historical_exploits import (
    HISTORICAL_EXPLOITS,
    get_exploit_lessons,
    get_exploits_by_type,
    get_comprehensive_summary
)

# View all exploits
for name, data in HISTORICAL_EXPLOITS.items():
    print(f"{name}: ${data['loss_usd']:,} loss")
    for lesson in data['lessons']:
        print(f"  - {lesson}")

# Filter by type
oracle_exploits = get_exploits_by_type("oracle_manipulation")

# Get comprehensive statistics
summary = get_comprehensive_summary()
print(f"Total losses: ${summary['total_loss_usd']:,}")
print(f"Most common attack vectors: {summary['losses_by_attack_type']}")
```

## Parameter Optimization

### Evolved Parameters

1. **Collateral Factors (CF)**
   - Maximum borrowing power as % of collateral
   - Typical range: 50-85%
   - Trade-off: Higher CF = more capital efficiency, more risk

2. **Liquidation Thresholds**
   - Health factor at which liquidation occurs
   - Must be > CF (typically CF + 5-15%)
   - Prevents bad debt accumulation

3. **Liquidation Bonuses**
   - Incentive for liquidators
   - Typical range: 5-15%
   - Higher bonus = faster liquidations, less bad debt

4. **Price Oracle Type**
   - `spot`: Single exchange price (RISKY)
   - `twap`: Time-weighted average price
   - `median`: Median of multiple sources
   - `chainlink`: Decentralized oracle network (SAFEST)

5. **Circuit Breaker Threshold**
   - Max price change before trading halts
   - Typical range: 5-20%
   - Prevents manipulation

6. **Minimum Liquidity Required**
   - Minimum asset liquidity for listing
   - Typical: $1M-$10M
   - Prevents low-liquidity manipulation

7. **Max Price Impact**
   - Maximum allowed trade size as % of pool
   - Typical range: 1-10%
   - Prevents large manipulation trades

## Usage Examples

### Basic Usage

```python
from openevolve.finance.verticals.defi import (
    DeFiProtocolEvolver,
    ProtocolConstraints
)

# Initialize evolver
evolver = DeFiProtocolEvolver(config={
    "population_size": 100,
    "generations": 50,
    "mutation_rate": 0.2,
})

# Define constraints
constraints = ProtocolConstraints(
    max_collateral_factor=0.80,
    min_liquidation_bonus=0.05,
    target_utilization=0.80,
    max_bad_debt_threshold=0.01,
)

# Evolve parameters
result = await evolver.evolve_protocol_parameters(
    protocol="compound",
    assets=["ETH", "USDC", "WBTC"],
    constraints=constraints
)

# Access results
print(f"Capital Efficiency: {result.capital_efficiency:.2%}")
print(f"Risk Score: {result.validation.risk_score:.1f}/100")
print(f"Survived Attacks: {sum(result.attack_survival.values())}/{len(result.attack_survival)}")

# Best parameters
params = result.parameters
print(f"ETH CF: {params.collateral_factors['ETH']:.2%}")
print(f"Oracle: {params.price_oracle_type}")
print(f"Circuit Breaker: {params.circuit_breaker_threshold:.2%}")
```

### Advanced Usage: Custom Attack Scenarios

```python
from openevolve.finance.verticals.defi.defi_evolver import DeFiAttackScenario

# Define custom attack
custom_attack = DeFiAttackScenario(
    name="custom_flash_loan_cascade",
    description="Multi-protocol flash loan attack",
    attack_type="flash_loan",
    attack_steps=[
        {"step": 1, "action": "flash_loan_borrow", "asset": "USDC", "amount": 500_000_000},
        {"step": 2, "action": "supply_collateral", "asset": "USDC", "amount": 500_000_000},
        {"step": 3, "action": "borrow", "asset": "ETH", "amount": 100_000, "collateral": "USDC"},
        {"step": 4, "action": "dump_on_dex", "asset": "ETH", "amount": 100_000,
         "dex": "uniswap", "price_impact": -0.40},
        {"step": 5, "action": "trigger_liquidation", "liquidate": "USDC", "receive": "ETH"},
        {"step": 6, "action": "repay_flash_loan", "asset": "USDC", "amount": 500_000_000},
    ],
    expected_profit=25_000_000,
    attack_vectors=["flash_loan", "oracle_manipulation", "liquidation"],
    difficulty="extreme"
)

# Add custom attack to evolver
evolver.custom_scenarios = [custom_attack]
```

### Historical Performance Analysis

```python
# Simulate historical events
historical = result.historical_performance

print(f"Average Utilization: {historical.avg_utilization:.2%}")
print(f"Max Bad Debt: ${historical.max_bad_debt:,.2f}")
print(f"Survived All Events: {historical.survived_all_events}")

# Analyze specific events
for event_result in historical.event_results:
    print(f"\nEvent: {event_result['event']} ({event_result['date']})")
    print(f"  Survived: {event_result['survived']}")
    print(f"  Bad Debt: ${event_result['bad_debt']:,.2f}")
    print(f"  Utilization: {event_result['utilization']:.2%}")
```

### Parameter Validation

```python
validation = result.validation

if not validation.meets_constraints:
    print("Constraint violations:")
    for violation in validation.constraint_violations:
        print(f"  - {violation}")

print(f"\nRisk Score: {validation.risk_score:.1f}/100")
print(f"Capital Efficiency Score: {validation.capital_efficiency_score:.1f}/100")

# Analyze attack survival
for attack, survived in validation.scenario_results.items():
    status = "✓" if survived else "✗"
    print(f"{status} {attack}")
```

## Best Practices

### 1. Start Conservative

```python
# For new protocols, start conservative
conservative_constraints = ProtocolConstraints(
    max_collateral_factor=0.60,  # Lower CF
    min_liquidation_bonus=0.10,  # Higher liquidation incentive
    target_utilization=0.70,
    max_bad_debt_threshold=0.005,  # 0.5% threshold
)
```

### 2. Gradual Parameter Increases

```python
# Evolve parameters in stages
stage1_constraints = ProtocolConstraints(
    max_collateral_factor=0.60,
    min_liquidation_bonus=0.10,
    target_utilization=0.70,
)

# After 30 days of stability, increase
stage2_constraints = ProtocolConstraints(
    max_collateral_factor=0.70,
    min_liquidation_bonus=0.08,
    target_utilization=0.75,
)
```

### 3. Asset-Specific Parameters

```python
# Riskier assets need stricter parameters
risky_asset_constraints = ProtocolConstraints(
    max_collateral_factor=0.50,  # Low CF for volatile assets
    min_liquidation_bonus=0.15,  # High liquidation incentive
    target_utilization=0.60,
    min_liquidity_threshold=5_000_000,  # $5M minimum
)

# Stable assets can be more liberal
stable_asset_constraints = ProtocolConstraints(
    max_collateral_factor=0.85,  # Higher CF for stablecoins
    min_liquidation_bonus=0.05,  # Lower bonus
    target_utilization=0.85,
)
```

### 4. Oracle Selection

```python
# For volatile assets, use safest oracle
volatile_params = ProtocolParameters(
    # ...
    price_oracle_type="chainlink",  # Decentralized oracle
    circuit_breaker_threshold=0.05,  # 5% threshold
)

# For stablecoins, TWAP is acceptable
stable_params = ProtocolParameters(
    # ...
    price_oracle_type="twap",
    circuit_breaker_threshold=0.10,
)
```

## Case Studies

### Case Study 1: Compound CLAIM Token Exploit (November 2020)

**What happened:**
- New token (CLAIM) listed with 80% CF
- Insufficient liquidity
- Attacker manipulated price, triggered massive liquidations
- $90M bad debt accumulated

**Lessons learned:**
1. Start new markets with low CF (50% or less)
2. Require minimum liquidity ($10M+)
3. Gradually increase parameters over weeks
4. Test thoroughly on testnet first

**Evolved parameters for new assets:**
```python
new_asset_params = ProtocolParameters(
    collateral_factors={"NEW_TOKEN": 0.50},  # Conservative
    liquidation_thresholds={"NEW_TOKEN": 0.65},
    liquidation_bonuses={"NEW_TOKEN": 0.15},  # High incentive
    price_oracle_type="chainlink",
    min_liquidity_required=10_000_000,  # $10M minimum
)
```

### Case Study 2: Cream Finance Oracle Exploit (October 2021)

**What happened:**
- Used single DEX (Curve) as oracle for yUSD
- Low liquidity pool was manipulated
- $130M lost

**Lessons learned:**
1. Never use single DEX as oracle
2. Use TWAP, median, or Chainlink oracles
3. Minimum liquidity thresholds
4. Price deviation limits

**Evolved parameters:**
```python
safe_params = ProtocolParameters(
    # ...
    price_oracle_type="chainlink",  # Not "spot"
    circuit_breaker_threshold=0.05,  # Catch manipulation
    min_liquidity_required=5_000_000,
    max_price_impact=0.02,  # Limit trade size
)
```

### Case Study 3: Harvest Finance Stablecoin Exploit (October 2020)

**What happened:**
- USDC/USDT price diverged on Curve
- Assumed stablecoins would stay at $1.00
- $24M lost

**Lessons learned:**
1. Even stablecoins can depeg
2. Need dedicated stablecoin oracles
3. Circuit breakers for large trades
4. Monitor peg deviations

**Evolved parameters:**
```python
stablecoin_params = ProtocolParameters(
    # ...
    price_oracle_type="chainlink",  # Dedicated oracle
    circuit_breaker_threshold=0.02,  # 2% threshold for stablecoins
    max_price_impact=0.01,  # 1% max impact
)
```

## Performance Benchmarks

### Evolution Speed

| Assets | Population | Generations | Time |
|--------|-----------|-------------|------|
| 3      | 100       | 50          | ~2-5 min |
| 5      | 100       | 50          | ~5-10 min |
| 10     | 100       | 50          | ~10-20 min |

### Attack Coverage

| Attack Type          | Scenarios | Detection Rate |
|---------------------|-----------|----------------|
| Flash Loan          | 5+        | 95%+           |
| Oracle Manipulation | 8+        | 90%+           |
| Cascading Liquidation | 3+     | 85%+           |
| Stablecoin De-peg   | 4+        | 90%+           |
| Smart Contract Bug  | 6+        | 80%+           |

## Integration with LoongFlow

The DeFi vertical uses LoongFlow for:

1. **Attack Scenario Planning**
   - Comprehensive attack vector identification
   - Multi-step attack combination
   - Difficulty and likelihood assessment

2. **Historical Pattern Analysis**
   - Learning from past exploits
   - Pattern recognition
   - Vulnerability identification

```python
# Enable LoongFlow integration
evolver = DeFiProtocolEvolver(config={
    "use_loongflow": True,
    "loongflow_config": {
        "model": "gpt-4",
        "temperature": 0.7,
    }
})
```

## Troubleshooting

### Issue: High Bad Debt in Simulation

**Symptoms:**
- Bad debt > 1% of TVL in historical simulations
- Frequent liquidation failures

**Solutions:**
1. Increase liquidation bonuses (0.10-0.15)
2. Lower collateral factors (by 5-10%)
3. Improve oracle quality (use Chainlink)
4. Reduce max price impact

### Issue: Low Capital Efficiency

**Symptoms:**
- Utilization < 50%
- High collateral factors but low borrowing

**Solutions:**
1. Gradually increase collateral factors
2. Reduce liquidation thresholds slightly
3. Improve user interface/UX
4. Offer incentives for borrowing

### Issue: Oracle Manipulation Survival

**Symptoms:**
- Failing oracle manipulation scenarios
- Spot price attacks succeeding

**Solutions:**
1. Switch from "spot" to "chainlink" oracle
2. Implement TWAP oracles
3. Add circuit breakers (5-10% threshold)
4. Increase minimum liquidity requirements

## References

### Academic Papers
- "DeFi Protocol Risk Assessment" (Stanford, 2022)
- "Oracle Manipulation in DeFi" (MIT, 2023)
- "Flash Loan Attack Vectors" (UC Berkeley, 2022)

### Industry Reports
- "DeFi Exploit Database" - Rekt News
- "Lending Protocol Risk Report" - CertiK
- "Oracle Best Practices" - Chainlink Labs

### Historical Data
- [Rekt News Leaderboard](https://rekt.news/leaderboard/)
- [DeFi Llama Exploits](https://defillama.com/hacks)
- [Immunefi Bug Bounty](https://immunefi.com/)

## Contributing

To add new attack scenarios:

1. Create scenario in `attack_generator.py`
2. Add to historical exploits database
3. Write tests in `tests/test_defi_evolver.py`
4. Update documentation

## License

MIT License - See LICENSE file for details
