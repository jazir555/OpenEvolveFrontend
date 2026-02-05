"""
M&A Deal Intelligence Platform - Usage Example

Demonstrates end-to-end M&A deal workflow automation.
"""

import asyncio
from datetime import datetime

from openevolve.agents.ma import (
    MADealPlatform,
    Deal,
    DealStage,
    DealPriority,
    TargetCompany,
    DealOutcome,
)


async def main():
    """Run complete M&A deal workflow example"""

    print("=" * 60)
    print("M&A Deal Intelligence Platform - Example Workflow")
    print("=" * 60)

    # 1. Initialize Platform
    print("\n1. Initializing M&A Deal Platform...")
    config = {
        "sourcer": {
            "market_criteria": [
                {
                    "industries": ["Technology", "Software", "SaaS"],
                    "size_range": (10, 500),  # $10M - $500M revenue
                    "growth_rate_min": 0.15,  # 15% minimum growth
                    "geographies": ["North America"],
                }
            ]
        },
        "valuation": {
            "market_data": {
                "multiples": {
                    "software": {
                        "ev_revenue": 5.0,
                        "ev_ebitda": 15.0,
                    }
                }
            }
        },
        "knowledge": {
            "success_patterns": {},
        }
    }

    platform = MADealPlatform(
        config=config,
        use_loongflow=False,  # Set to True if LoongFlow available
        workspace_dir="./example_ma_workspace",
    )

    print("   [OK] Platform initialized")

    # 2. Create Target Company (simulating sourcing result)
    print("\n2. Creating Target Company...")
    target = TargetCompany(
        company_id="target_example_001",
        name="CloudTech Solutions Inc.",
        industry="Software",
        sector="Technology",
        revenue=150.0,  # $150M revenue
        ebitda=30.0,  # $30M EBITDA (20% margin)
        employees=400,
        growth_rate=0.28,  # 28% growth rate
        description="Leading cloud-based ERP software for mid-market companies",
        headquarters="Austin, TX",
        website="www.cloudtechexample.com",
    )

    print(f"   [OK] Target: {target.name}")
    print(f"     - Revenue: ${target.revenue:.1f}M")
    print(f"     - Growth: {target.growth_rate:.1%}")
    print(f"     - Employees: {target.employees}")

    # 3. Analyze Strategic Fit
    print("\n3. Analyzing Strategic Fit...")
    fit = await platform.sourcer.analyze_strategic_fit(target)
    target.strategic_fit = fit

    print(f"   [OK] Overall Score: {fit.overall_score:.1%}")
    print(f"     - Strategic Alignment: {fit.strategic_alignment:.1%}")
    print(f"     - Cultural Fit: {fit.cultural_fit:.1%}")
    print(f"     - Technology Compatibility: {fit.technology_compatibility:.1%}")
    print(f"     - Market Expansion: {fit.market_expansion:.1%}")
    print(f"     - Synergies Identified: {len(fit.synergies)}")
    for synergy in fit.synergies:
        print(f"       * {synergy.synergy_type.value}: "
              f"${synergy.estimated_value:.1f}M ({synergy.time_to_realize} months)")

    # 4. Create Deal
    print("\n4. Creating Deal...")
    deal = Deal(
        deal_id="deal_example_001",
        target_company=target,
        stage=DealStage.SOURCING,
        priority=DealPriority.HIGH,
        created_at=datetime.utcnow(),
    )

    platform.deals[deal.deal_id] = deal
    print(f"   [OK] Deal created: {deal.deal_id}")

    # 5. Initiate Due Diligence
    print("\n5. Initiating Due Diligence...")
    diligence_report = await platform.initiate_diligence(
        deal.deal_id,
        diligence_depth="standard",
    )

    print(f"   [OK] Diligence Complete")
    print(f"     - Recommendation: {diligence_report.recommendation.upper()}")
    print(f"     - Confidence: {diligence_report.confidence:.1%}")
    print(f"     - Financial Health: {diligence_report.financial_health}")
    print(f"     - Legal Compliance: {diligence_report.legal_compliance}")
    print(f"     - Risks: {len(diligence_report.risks)}")
    print(f"     - Red Flags: {len(diligence_report.red_flags)}")
    print(f"     - Opportunities: {len(diligence_report.opportunities)}")

    # Check if we should proceed
    if diligence_report.recommendation == "reject":
        print("\n   [WARN] Deal rejected based on due diligence")
        return

    # 6. Analyze Deal (Valuation, Structure, Integration)
    print("\n6. Analyzing Deal...")
    valuation, structure, integration_plan = await platform.analyze_deal(deal.deal_id)

    print(f"   [OK] Valuation: ${valuation.implied_value:.1f}M")
    print(f"     - Range: ${valuation.valuation_range[0]:.1f}M - "
          f"${valuation.valuation_range[1]:.1f}M")
    print(f"     - Methods: {len(valuation.methods)}")
    for method in valuation.methods:
        print(f"       * {method.method}: ${method.value:.1f}M "
              f"(confidence: {method.confidence:.1%})")
    print(f"     - Synergy Value: ${valuation.synergy_value:.1f}M")

    print(f"\n   [OK] Deal Structure:")
    print(f"     - Total Value: ${structure.total_value:.1f}M")
    print(f"     - Cash: ${structure.cash_component:.1f}M "
          f"({structure.cash_component/structure.total_value:.0%})")
    print(f"     - Stock: ${structure.stock_component:.1f}M "
          f"({structure.stock_component/structure.total_value:.0%})")
    print(f"     - Earnout: ${structure.earnout:.1f}M "
          f"({structure.earnout/structure.total_value:.0%})")
    print(f"     - Tax Efficiency: {structure.tax_efficiency:.1%}")
    print(f"     - Efficiency Score: {structure.efficiency_score:.1%}")

    print(f"\n   [OK] Integration Plan:")
    print(f"     - Day 1 Items: {len(integration_plan.day_1_plan)}")
    print(f"     - 30-Day Milestones: {len(integration_plan.first_30_days)}")
    print(f"     - 90-Day Milestones: {len(integration_plan.first_90_days)}")
    print(f"     - Synergy Plans: {len(integration_plan.synergy_plans)}")
    print(f"     - Workstreams: {len(integration_plan.workstreams)}")

    # 7. Generate Recommendation
    print("\n7. Generating Recommendation...")
    recommendation = await platform.generate_recommendation(deal.deal_id)

    print(f"   [OK] Recommendation: {recommendation['recommendation'].upper()}")
    print(f"     - Confidence: {recommendation['confidence']:.1%}")
    print(f"     - Rationale: {recommendation['rationale']}")

    # 8. Prepare Negotiation Strategy
    print("\n8. Preparing Negotiation Strategy...")
    strategy = await platform.prepare_negotiation(deal.deal_id)

    print(f"   [OK] Strategy: {strategy.approach}")
    print(f"     - Our BATNA: {strategy.batna.description}")
    print(f"       Value: ${strategy.batna.value:.1f}M, "
          f"Probability: {strategy.batna.probability:.1%}")
    print(f"     - Their BATNA: {strategy.their_batna.description}")
    print(f"       Value: ${strategy.their_batna.value:.1f}M, "
          f"Probability: {strategy.their_batna.probability:.1%}")
    print(f"     - Leverage: {strategy.leverage_assessment}")
    print(f"     - Must Haves: {len(strategy.must_haves)}")
    for item in strategy.must_haves[:3]:
        print(f"       * {item}")
    print(f"     - Tradeables: {len(strategy.tradeables)}")
    for item in strategy.tradeables[:3]:
        print(f"       * {item}")

    # 9. Close Deal
    print("\n9. Closing Deal...")
    final_value = structure.total_value
    await platform.close_deal(deal.deal_id, final_value=final_value)

    print(f"   [OK] Deal closed at ${final_value:.1f}M")
    print(f"     - Stage: {deal.stage.value}")

    # 10. Start Integration
    print("\n10. Starting Integration...")
    active_plan = await platform.start_integration(deal.deal_id)

    print(f"   [OK] Integration started")
    print(f"     - Steering Committee: {len(active_plan.steering_committee)} members")
    print(f"     - Communication Plan: {len(active_plan.communication_plan)} items")
    print(f"     - Retention Plan: {len(active_plan.retention_plan)} items")

    # 11. Record Outcome (simulated successful completion)
    print("\n11. Recording Deal Outcome...")

    # Simulate waiting for integration to complete
    print("    (Simulating 12-month integration period...)")

    outcome = DealOutcome(
        deal_id=deal.deal_id,
        outcome="completed",
        closed_date=datetime.utcnow(),
        final_value=final_value,
        actual_vs_expected=1.05,  # 5% over expectations
        synergies_realized=21.0,
        synergies_expected=23.0,
        synergy_realization_rate=0.91,
        integration_success=True,
        integration_timeline_adherence=0.95,
        key_success_factors=[
            "Strong leadership alignment",
            "Effective cultural integration",
            "Key talent retention",
            "Technology integration success",
        ],
        key_challenges=[
            "Longer than expected CRM integration",
            "Some customer contract transitions delayed",
        ],
        lessons_learned=[
            "Start IT integration earlier in the process",
            "Dedicate more resources to customer communication",
            "Focus on quick wins to build momentum",
        ],
        would_repeat=True,
        recommendation_for_future="Similar targets in adjacent markets "
                                   "show strong potential",
    )

    await platform.record_outcome(deal.deal_id, outcome)

    print(f"   [OK] Outcome recorded")
    print(f"     - Final Value: ${outcome.final_value:.1f}M")
    print(f"     - Actual vs Expected: {outcome.actual_vs_expected:.1%}")
    print(f"     - Synergy Realization: {outcome.synergy_realization_rate:.1%}")
    print(f"     - Integration Success: {outcome.integration_success}")
    print(f"     - Would Repeat: {outcome.would_repeat}")

    # 12. Show Learning
    print("\n12. Platform Learning...")
    patterns = await platform.knowledge.get_success_patterns()

    print(f"   [OK] Success Patterns Identified: {len(patterns)}")
    for pattern_name, pattern in patterns.items():
        print(f"     - {pattern.pattern}")
        print(f"       Success Rate: {pattern.success_rate:.1%}, "
              f"Evidence: {pattern.evidence_count} deals")

    # 13. Pipeline Summary
    print("\n13. Pipeline Summary...")
    summary = await platform.get_pipeline_summary()

    print(f"   [OK] Total Deals: {summary['total_deals']}")
    print(f"   [OK] By Stage:")
    for stage, count in summary['by_stage'].items():
        if count > 0:
            print(f"     - {stage}: {count}")
    print(f"   [OK] Conversion Rate: {summary['conversion_rate']:.1f}")
    print(f"   [OK] Average Deal Size: ${summary['average_deal_size']:.1f}M")

    print("\n" + "=" * 60)
    print("M&A Deal Workflow Complete!")
    print("=" * 60)

    # Cleanup
    import shutil
    try:
        shutil.rmtree("./example_ma_workspace")
    except:
        pass


if __name__ == "__main__":
    asyncio.run(main())
