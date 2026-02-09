"""
SSV Multi-Cluster Insolvency Demo (Python)
Demonstrates how multiple bankrupt clusters compound the debt.
"""

def run_demo():
    print(">>> SSV POC 2: Multi-Cluster Insolvency (Python Demo)")
    
    # Setup
    victim_deposit = 10000
    small_deposits = [100, 50, 25] # 3 Small clusters
    
    pool_assets = victim_deposit + sum(small_deposits)
    
    print(f"[INIT] Pool Assets: {pool_assets}")
    
    # Simulation Parameters
    blocks = 150
    op_fee = 1
    dao_fee = 0.5
    
    total_virtual_debt = 0
    
    # Process Clusters
    for i, deposit in enumerate(small_deposits):
        # Calculate when they go bankrupt
        burn_rate = op_fee + dao_fee
        bankruptcy_block = int(deposit / burn_rate)
        
        # Calculate Unbacked Fees (Post-Bankruptcy)
        unbacked_blocks = max(0, blocks - bankruptcy_block)
        
        # Operator Earnings (Unbacked)
        op_unbacked = unbacked_blocks * op_fee
        
        # DAO Earnings (Unbacked)
        dao_unbacked = unbacked_blocks * dao_fee
        
        cluster_debt = op_unbacked + dao_unbacked
        total_virtual_debt += cluster_debt
        
        print(f"[CLUSTER {i+1}] Bankrupt at block {bankruptcy_block}. Unbacked Blocks: {unbacked_blocks}")
        print(f"            Generated Virtual Debt: {cluster_debt}")

    print(f"[TOTAL] Global Virtual Debt: {total_virtual_debt}")
    
    # Withdrawals
    pool_assets -= total_virtual_debt # Operators/DAO withdraw "earnings"
    
    print(f"[FINAL] Pool Assets Remaining: {pool_assets}")
    
    if pool_assets < victim_deposit:
        print(f"CRITICAL: Victim Lost {victim_deposit - pool_assets}!")
        print("Bank Run Logic Confirmed.")

if __name__ == "__main__":
    run_demo()
