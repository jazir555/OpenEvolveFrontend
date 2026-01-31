# M&A Deal Intelligence Platform

## Quick Start

```python
import asyncio
from openevolve.agents.ma import MADealPlatform, Deal, TargetCompany, DealStage

async def main():
    # Initialize platform
    platform = MADealPlatform(config={...})

    # Create deal
    target = TargetCompany(
        company_id="target_001",
        name="Tech Target Inc.",
        industry="Software",
        revenue=100.0,
        ebitda=20.0,
        employees=300,
        growth_rate=0.25,
    )

    deal = Deal(
        deal_id="deal_001",
        target_company=target,
        stage=DealStage.SOURCING,
    )

    # Process deal
    await platform.initiate_diligence(deal.deal_id)
    valuation, structure, integration = await platform.analyze_deal(deal.deal_id)
    recommendation = await platform.generate_recommendation(deal.deal_id)

    print(f"Recommendation: {recommendation['recommendation']}")
    print(f"Value: ${valuation.implied_value:.1f}M")

asyncio.run(main())
```

## Key Features

- **Continuous Deal Sourcing**: Automated market scanning and target identification
- **Due Diligence Automation**: Comprehensive checklists and risk analysis
- **Multi-Method Valuation**: DCF, comparables, precedent transactions
- **Deal Structure Optimization**: Tax-efficient deal structuring
- **Negotiation Support**: BATNA analysis and strategy recommendations
- **Integration Planning**: Post-merger integration roadmaps
- **Continuous Learning**: Learns from outcomes to improve recommendations

## Installation

```bash
pip install openevolve[ma]
```

## Components

| Component | Purpose |
|-----------|---------|
| `MADealPlatform` | Main orchestrator |
| `DealSourcer` | Market scanning and target identification |
| `DiligenceAssistant` | Due diligence automation |
| `ValuationEngine` | Comprehensive valuation |
| `StructureOptimizer` | Deal structure optimization |
| `NegotiationAdvisor` | Negotiation strategy |
| `IntegrationPlanner` | Post-merger integration |
| `DealKnowledgeManager` | Learning and patterns |

## Example Output

```
M&A Deal Intelligence Platform - Example Workflow
============================================================

1. Initializing M&A Deal Platform...
   ✓ Platform initialized

2. Creating Target Company...
   ✓ Target: CloudTech Solutions Inc.
     - Revenue: $150.0M
     - Growth: 28.0%
     - Employees: 400

3. Analyzing Strategic Fit...
   ✓ Overall Score: 75.0%
     - Strategic Alignment: 80.0%
     - Synergies Identified: 2
       • revenue: $15.0M (18 months)
       • cost: $8.0M (12 months)

5. Initiating Due Diligence...
   ✓ Diligence Complete
     - Recommendation: PROCEED
     - Confidence: 80.0%
     - Risks: 3
     - Red Flags: 0

6. Analyzing Deal...
   ✓ Valuation: $325.0M
     - Range: $260.0M - $390.0M
     - Synergy Value: $23.0M

   ✓ Deal Structure:
     - Total Value: $325.0M
     - Cash: 60%
     - Stock: 30%
     - Earnout: 10%

7. Generating Recommendation...
   ✓ Recommendation: PROCEED
     - Confidence: 75.0%

M&A Deal Workflow Complete!
============================================================
```

## Documentation

Full documentation: [docs/agents/ma_platform.md](../docs/agents/ma_platform.md)

## Testing

```bash
pytest openevolve/tests/agents/test_ma_platform.py -v
```

## License

Apache 2.0
