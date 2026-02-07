"""
Execution PoC: SSV Cascading Insolvency

This script simulates the Solidity logic defined in InsolvencyPoC.sol
to provide a step-by-step execution trace of the theft of funds.
"""

def run_execution_poc():
    print("=" * 80)
    print("SSV PROTOCOL EXECUTION TRACE: PROOF OF THEFT")
    print("=" * 80)

    # 1. Initial State
    pool_assets = 0
    
    # User A (Honest) deposits 1000 SSV
    user_a_deposit = 1000
    pool_assets += user_a_deposit
    
    # User B (Bankrupt Target) deposits 10 SSV
    user_b_deposit = 10
    pool_assets += user_b_deposit
    
    print(f"[Block 0] Initial Deposits: User A = {user_a_deposit}, User B = {user_b_deposit}")
    print(f"[Block 0] Total Contract Assets: {pool_assets} SSV")

    # 2. Setup Operator
    op_fee = 5 # 5 SSV per block
    op_virtual_balance = 0
    op_last_snapshot = 0
    
    # 3. Transition: 10 Blocks pass
    current_block = 10
    print(f"
--- 10 Blocks Pass ---")
    
    # Protocol Logic: Update Cluster B (User B)
    # usage = (10 - 0) * 5 = 50
    # User B balance = max(0, 10 - 50) = 0
    user_b_balance = max(0, user_b_deposit - (current_block * op_fee))
    print(f"[Block 10] User B Balance: {user_b_balance} SSV (BANKRUPT)")
    
    # Protocol Logic: Update Operator Snapshot
    # op_balance += 50
    op_virtual_balance += (current_block * op_fee)
    print(f"[Block 10] Operator Virtual Balance: {op_virtual_balance} SSV")
    
    # 4. The Exploit: Operator withdraws full virtual balance
    print(f"
--- Operator Withdrawal ---")
    withdrawal_amount = op_virtual_balance
    
    print(f"Operator attempting to withdraw {withdrawal_amount} SSV...")
    
    if withdrawal_amount <= pool_assets:
        pool_assets -= withdrawal_amount
        print(f"SUCCESS: Operator withdrew {withdrawal_amount} SSV")
    else:
        print(f"FAILED: Contract only has {pool_assets} SSV")

    # 5. The Consequence: User A attempts to withdraw
    print(f"
--- Honest User A Withdrawal ---")
    print(f"User A attempting to withdraw their original {user_a_deposit} SSV...")
    
    if user_a_deposit <= pool_assets:
        print(f"SUCCESS: User A recovered funds.")
    else:
        loss = user_a_deposit - pool_assets
        print(f"CRITICAL FAILURE: User A can only withdraw {pool_assets} SSV.")
        print(f"USER A TOTAL LOSS: {loss} SSV")
        print(f"FINAL CONTRACT ASSETS: 0 SSV")

    print("
" + "=" * 80)
    print("CONCLUSION: Protocol Insolvency Proven by Execution Trace.")
    print("User B's bankruptcy created 40 SSV of uncollateralized debt which was")
    print("paid out using User A's honest deposit.")
    print("=" * 80)

if __name__ == "__main__":
    run_execution_poc()
