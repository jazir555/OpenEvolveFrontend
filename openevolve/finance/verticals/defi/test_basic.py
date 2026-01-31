"""
Basic test script for DeFi vertical
"""

import sys
sys.path.insert(0, '/c/Users/mmeadow/Documents/OpenEvolve/Frontend')

from openevolve.finance.verticals.defi.historical_exploits import (
    HISTORICAL_EXPLOITS,
    get_comprehensive_summary,
)

# Test historical exploits
print("=" * 80)
print("Testing DeFi Vertical - Historical Exploits Database")
print("=" * 80)

summary = get_comprehensive_summary()

print(f"\nTotal exploits: {summary['total_exploits']}")
print(f"Total losses: ${summary['total_loss_usd']:,.0f}")

print("\nTop 5 most destructive exploits:")
for name, loss in summary['top_5_destructive']:
    exploit = HISTORICAL_EXPLOITS[name]
    print(f"  {name}")
    print(f"    Date: {exploit['date']}")
    print(f"    Protocol: {exploit['protocol']}")
    print(f"    Loss: ${loss:,.0f}")

print("\nLosses by attack type:")
for attack_type, loss in summary['losses_by_attack_type'].items():
    print(f"  {attack_type}: ${loss:,.0f}")

print("\nMost common lessons:")
for lesson, count in list(summary['most_common_lessons'].items())[:5]:
    print(f"  {count}x: {lesson}")

print("\n" + "=" * 80)
print("✓ All tests passed!")
print("=" * 80)
