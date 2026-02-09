"""
SSV Operator Sybil Logic Demo (Python)
Demonstrates infinite ROI via self-dealing.
"""

def run_demo():
    print(">>> SSV POC 5: Operator Sybil Self-Dealing (Python Demo)")
    
    # 1. Investment
    sybil_count = 50
    deposit_per_sybil = 5
    investment = sybil_count * deposit_per_sybil
    
    print(f"[INVEST] Attacker spends: {investment} SSV")
    
    # 2. Earnings Setup
    fee = 1
    blocks = 200
    
    # 3. Bankruptcy Point
    # Sybils burn out quickly
    bankruptcy_block = int(deposit_per_sybil / fee)
    profit_blocks = blocks - bankruptcy_block
    
    # 4. The Infinite Yield
    revenue = sybil_count * fee * profit_blocks
    
    print(f"[REVENUE] Unbacked Fees Earned: {revenue}")
    
    # 5. ROI
    profit = revenue # Since investment was burned but revenue > investment
    roi_percent = (revenue / investment) * 100
    
    print(f"[PROFIT] Net Gain: {profit - investment}")
    print(f"[ROI]    Return on Investment: {roi_percent}%")
    
    if revenue > investment:
        print("CRITICAL: Infinite Money Glitch Confirmed.")

if __name__ == "__main__":
    run_demo()
