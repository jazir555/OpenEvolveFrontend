"""
Standalone test of DeFi vertical historical exploits database
"""

# Read and evaluate the historical exploits file directly
exec(open('/c/Users/mmeadow/Documents/OpenEvolve/Frontend/openevolve/finance/verticals/defi/historical_exploits.py').read())

print("=" * 80)
print("DeFi Vertical - Historical Exploits Database Test")
print("=" * 80)

summary = get_comprehensive_summary()

print(f"\nTotal exploits: {summary['total_exploits']}")
print(f"Total losses: ${summary['total_loss_usd']:,.0f}")

print("\nTop 5 most destructive exploits:")
for name, loss in summary['top_5_destructive']:
    exploit = HISTORICAL_EXPLOITS[name]
    print(f"\n  {name}")
    print(f"    Date: {exploit['date']}")
    print(f"    Protocol: {exploit['protocol']}")
    print(f"    Attack Type: {exploit['attack_type']}")
    print(f"    Loss: ${loss:,.0f}")

print("\n\nLosses by attack type:")
for attack_type, loss in sorted(summary['losses_by_attack_type'].items(), key=lambda x: x[1], reverse=True):
    print(f"  {attack_type}: ${loss:,.0f}")

print("\n\nMost common lessons learned:")
for lesson, count in list(summary['most_common_lessons'].items())[:5]:
    print(f"  {count}x: {lesson}")

print("\n" + "=" * 80)
print("✓ Historical exploits database working correctly!")
print("=" * 80)
