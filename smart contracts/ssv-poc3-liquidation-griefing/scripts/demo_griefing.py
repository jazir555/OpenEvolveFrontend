"""
SSV Liquidation Griefing Logic Demo (Python)
Demonstrates the maximized debt accumulation via delayed liquidation.
"""

def run_demo():
    print(">>> SSV POC 3: Liquidation Griefing (Python Demo)")
    
    # 1. Normal Liquidation Scenario
    print("\n--- SCENARIO 1: Perfect Liquidation ---")
    deposit = 100
    fee = 1
    liquidation_threshold = 80 # Block where it *should* liquidate
    
    unbacked_blocks_normal = 0
    print(f"Cluster Liquidated at Block {liquidation_threshold}")
    print(f"Unbacked Debt: {unbacked_blocks_normal}")
    
    # 2. Griefing Scenario
    print("\n--- SCENARIO 2: Griefing Attack ---")
    grief_delay = 200
    actual_liquidation = liquidation_threshold + grief_delay
    
    # The Gap: While delayed, fees accrue uncollaterally
    unbacked_blocks_griefed = grief_delay
    unbacked_fees = unbacked_blocks_griefed * fee
    
    print(f"Attacker Delays Liquidation by {grief_delay} Blocks!")
    print(f"Actual Liquidation Block: {actual_liquidation}")
    print(f"Unbacked Debt Created: {unbacked_fees}")
    
    # 3. Impact
    victim_assets = 10000
    pool = victim_assets + deposit
    
    # Withdrawal
    pool -= unbacked_fees
    
    print(f"\n[FINAL] Victim Assets Remaining: {pool}")
    if pool < victim_assets:
        loss = victim_assets - pool
        print(f"CRITICAL: Griefing stole {loss} SSV from honest users!")

if __name__ == "__main__":
    run_demo()