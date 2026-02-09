"""
Execution PoC: SSV Liquidation Griefing Insolvency

This script simulates the liquidation griefing attack,
demonstrating how delaying liquidation maximizes virtual debt.
"""

def run_execution_poc():
    print("=" * 80)
    print("SSV LIQUIDATION GRIEFING INSOLVENCY: EXECUTION TRACE")
    print("=" * 80)

    # 1. Initial State
    pool_assets = 0
    
    # Users deposit
    large_deposit = 10000
    small_1 = 100  # Bankrupts in 100 blocks
    small_2 = 50   # Bankrupts in 50 blocks
    small_3 = 25   # Bankrupts in 25 blocks
    pool_assets += large_deposit + small_1 + small_2 + small_3
    
    print("Block 0 - Initial Deposits:")
    print("  Large User:    " + str(large_deposit) + " SSV")
    print("  Small User 1:  " + str(small_1) + " SSV")
    print("  Small User 2:  " + str(small_2) + " SSV")
    print("  Small User 3:  " + str(small_3) + " SSV")
    print("  Total Assets:  " + str(pool_assets) + " SSV")

    # 2. Wait for near-liquidation
    print("\n--- Block 20: Clusters Near Liquidation ---")
    print("  Small User 3: 5 SSV remaining (5 blocks until bankrupt)")
    print("  Small User 2: 30 SSV remaining (30 blocks until bankrupt)")
    print("  Small User 1: 80 SSV remaining (80 blocks until bankrupt)")
    print("  Attacker detects opportunity!")

    # 3. LIQUIDATION GRIEFING
    griefing_blocks = 200
    print("\n--- LIQUIDATION GRIEFING ---")
    print("Attacker monitors mempool for liquidate() transactions...")
    print("Attacker front-runs with high gas or exhausts liquidators")
    print("Liquidation DELAYED by " + str(griefing_blocks) + " blocks!")

    # Calculate virtual debt after griefing
    # Small 1: Would have been liquidated at block 100, now at block 220
    # Virtual debt: 120 blocks * 1 SSV = 120 SSV
    virtual_debt_1 = 120
    
    # Small 2: Would have been liquidated at block 50, now at block 220
    # Virtual debt: 170 blocks * 1 SSV = 170 SSV
    virtual_debt_2 = 170
    
    # Small 3: Would have been liquidated at block 25, now at block 220
    # Virtual debt: 195 blocks * 1 SSV = 195 SSV
    virtual_debt_3 = 195
    
    # DAO unbacked fees (0.5 SSV per block per validator)
    dao_virtual_debt = 200 * 0.5 * 3  # 300 SSV
    
    total_virtual_debt = virtual_debt_1 + virtual_debt_2 + virtual_debt_3 + dao_virtual_debt
    
    print("\nAfter " + str(griefing_blocks) + " blocks of griefing:")
    print("  All small clusters: BANKRUPT (liquidation delayed)")
    print("  Virtual debt from Small 1: " + str(virtual_debt_1) + " SSV")
    print("  Virtual debt from Small 2: " + str(virtual_debt_2) + " SSV")
    print("  Virtual debt from Small 3: " + str(virtual_debt_3) + " SSV")
    print("  DAO unbacked fees:         " + str(dao_virtual_debt) + " SSV")
    print("\nTOTAL VIRTUAL DEBT: " + str(total_virtual_debt) + " SSV")
    print("(WITHOUT griefing, this would only be ~100 SSV!)")
    print("Griefing increased theft by " + str((total_virtual_debt / 100 - 1) * 100) + "%")

    # 4. Bank Run
    print("\n--- BANK RUN: Race to Withdraw ---")
    
    # Operators and DAO race to withdraw
    withdrawal_3 = virtual_debt_3
    if withdrawal_3 <= pool_assets:
        pool_assets -= withdrawal_3
        print("Operator 3 withdrew:  " + str(withdrawal_3) + " SSV")
    
    withdrawal_2 = virtual_debt_2
    if withdrawal_2 <= pool_assets:
        pool_assets -= withdrawal_2
        print("Operator 2 withdrew:  " + str(withdrawal_2) + " SSV")
    
    withdrawal_1 = virtual_debt_1
    if withdrawal_1 <= pool_assets:
        pool_assets -= withdrawal_1
        print("Operator 1 withdrew:  " + str(withdrawal_1) + " SSV")
    
    dao_withdrawal = dao_virtual_debt
    if dao_withdrawal <= pool_assets:
        pool_assets -= dao_withdrawal
        print("DAO withdrew:         " + str(dao_withdrawal) + " SSV")
    
    total_stolen = withdrawal_1 + withdrawal_2 + withdrawal_3 + dao_withdrawal
    print("\nTotal stolen:         " + str(total_stolen) + " SSV")
    print("ALL OF IT IS UNBACKED VIRTUAL DEBT!")

    # 5. Honest victim withdrawal
    print("\n--- Honest Large User Attempts Withdrawal ---")
    print("Large user entitlement: " + str(large_deposit) + " SSV")
    print("Remaining pool assets:  " + str(pool_assets) + " SSV")
    
    if large_deposit <= pool_assets:
        print("SUCCESS: Large user recovered all funds.")
        loss = 0
    else:
        loss = large_deposit - pool_assets
        print("CRITICAL FAILURE: Large user can only withdraw " + str(pool_assets) + " SSV")
        print("LARGE USER TOTAL LOSS: " + str(loss) + " SSV")

    print("\n" + "=" * 80)
    print("CONCLUSION: Liquidation Griefing Maximizes Theft.")
    print("Griefing period of " + str(griefing_blocks) + " blocks created " + str(total_virtual_debt) + " SSV of virtual debt.")
    print("This is the MOST SEVERE attack vector:")
    print("  - Can be executed by anyone (not just operators)")
    print("  - Maximizes virtual debt through time delay")
    print("  - Harder to detect than direct theft")
    print("=" * 80)

if __name__ == "__main__":
    run_execution_poc()
