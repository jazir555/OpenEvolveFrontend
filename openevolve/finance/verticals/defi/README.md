# DeFi Vertical - Protocol Risk Evolution

Evolve DeFi lending protocol parameters that survive historical exploits and black swan attacks.

## Features

- **Attack Simulation**: Test against 20+ realistic attack scenarios
- **Historical Learning**: Learn from $2B+ in historical DeFi exploits
- **Parameter Optimization**: Evolve robust collateral factors, liquidation thresholds, and oracle settings
- **Risk Assessment**: Comprehensive risk scoring and validation
- **Capital Efficiency**: Balance safety with utilization

## Quick Start

```python
import asyncio
from openevolve.finance.verticals.defi import (
    DeFiProtocolEvolver,
    ProtocolConstraints
)

async def main():
    # Initialize evolver
    evolver = DeFiProtocolEvolver(config={
        "population_size": 100,
        "generations": 50,
    })

    # Define constraints
    constraints = ProtocolConstraints(
        max_collateral_factor=0.80,
        min_liquidation_bonus=0.05,
        target_utilization=0.80,
    )

    # Evolve parameters
    result = await evolver.evolve_protocol_parameters(
        protocol="compound",
        assets=["ETH", "USDC", "WBTC"],
        constraints=constraints
    )

    # Print results
    print(f"Capital Efficiency: {result.capital_efficiency:.2%}")
    print(f"Risk Score: {result.validation.risk_score:.1f}/100")
    print(f"Survived Attacks: {sum(result.attack_survival.values())}/{len(result.attack_survival)}")

    print("\nOptimal Parameters:")
    print(f"  ETH Collateral Factor: {result.parameters.collateral_factors['ETH']:.2%}")
    print(f"  Oracle Type: {result.parameters.price_oracle_type}")
    print(f"  Circuit Breaker: {result.parameters.circuit_breaker_threshold:.2%}")

if __name__ == "__main__":
    asyncio.run(main())
```

## Installation

```bash
# Clone repository
git clone https://github.com/openevolve/finance.git
cd finance

# Install dependencies
pip install -r requirements.txt

# Install DeFi vertical
pip install -e openevolve/finance/verticals/defi
```

## Documentation

See [docs/DEFI.md](docs/DEFI.md) for comprehensive documentation including:

- Risk Framework
- Attack Scenario Methodology
- Parameter Optimization Guide
- Usage Examples
- Best Practices
- Case Studies
- Performance Benchmarks

## Testing

```bash
# Run all tests
pytest openevolve/finance/verticals/defi/tests/

# Run specific test
pytest openevolve/finance/verticals/defi/tests/test_defi_evolver.py::test_end_to_end_evolution -v

# Run with coverage
pytest --cov=openevolve.finance.verticals.defi openevolve/finance/verticals/defi/tests/
```

## Supported Protocols

- Compound
- Aave
- Venus
- MakerDAO
- Liquity
- Custom lending protocols

## Attack Scenarios

- Flash Loan Attacks
- Oracle Manipulation (wash trading, TWAP manipulation)
- Cascading Liquidations (systemic risk)
- Stablecoin De-pegs (UST, USDD, etc.)
- Smart Contract Bugs (reentrancy, overflow)
- Historical Exploits (bZX, Cream Finance, etc.)

## Parameters Evolved

1. **Collateral Factors** - Maximum borrowing power
2. **Liquidation Thresholds** - Health factor limits
3. **Liquidation Bonuses** - Liquidator incentives
4. **Price Oracle Type** - Spot, TWAP, Median, Chainlink
5. **Circuit Breaker Threshold** - Price change limits
6. **Minimum Liquidity** - Asset listing requirements
7. **Max Price Impact** - Trade size limits

## Performance

| Configuration | Time | Attacks Tested | Accuracy |
|--------------|------|----------------|----------|
| 3 assets, 100 pop, 50 gen | ~3 min | 20+ | 95%+ |
| 5 assets, 100 pop, 50 gen | ~7 min | 25+ | 93%+ |
| 10 assets, 100 pop, 50 gen | ~15 min | 30+ | 90%+ |

## Citation

```bibtex
@software{openevolve_defi_2024,
  title={DeFi Protocol Risk Evolution System},
  author={OpenEvolve Finance Team},
  year={2024},
  url={https://github.com/openevolve/finance}
}
```

## License

MIT License - See LICENSE file for details

## Contributing

Contributions welcome! Please see CONTRIBUTING.md for guidelines.

## Contact

- GitHub Issues: https://github.com/openevolve/finance/issues
- Discord: https://discord.gg/openevolve
- Email: defi@openevolve.finance
