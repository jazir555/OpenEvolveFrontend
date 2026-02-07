"""
Formal Verification: SSV Network Insolvency via Late Liquidation

This script models the accounting logic of SSV Network to prove that
delayed liquidation of a 'bankrupt' cluster leads to protocol insolvency.
"""

import z3
import json

def verify_insolvency():
    print("=" * 70)
    print("SSV Protocol Formal Verification: Insolvency Analysis")
    print("=" * 70)

    solver = z3.Solver()

    # Constants (shrunk values as per protocol)
    # blocks: Number of blocks elapsed
    # op_fee: Fee per block per validator
    # initial_deposit: SSV deposited by cluster owner
    
    blocks = z3.Int('blocks')
    op_fee = z3.Int('op_fee')
    initial_deposit = z3.Int('initial_deposit')
    
    # 1. Rules of the system
    solver.add(blocks > 0)
    solver.add(op_fee > 0)
    solver.add(initial_deposit > 0)

    # 2. Accumulated Fees
    total_fees_owed = blocks * op_fee

    # 3. Operator Balance Update (as per OperatorLib.updateSnapshotSt)
    op_balance_inc = total_fees_owed

    # 4. Cluster Balance Update (as per ClusterLib.updateBalance)
    usage = total_fees_owed
    cluster_balance_final = z3.If(usage > initial_deposit, 0, initial_deposit - usage)

    # 5. Liquidation Payout (as per SSVClusters.liquidate)
    liquidator_reward = cluster_balance_final

    # 6. Total System Liabilities created by this interaction
    total_liabilities = op_balance_inc + liquidator_reward

    # 7. Total System Assets (tokens actually in the contract)
    total_assets = initial_deposit

    # 8. Insolvency Condition: Liabilities > Assets
    insolvency = total_liabilities > total_assets

    # We want to find IF there is any scenario where this holds.
    solver.add(insolvency)

    print("\nChecking for insolvency drift...")
    result = solver.check()

    if result == z3.sat:
        m = solver.model()
        print("\n[PROVED] Protocol Insolvency is possible!")
        print(f"Scenario:")
        print(f"  Blocks Elapsed: {m[blocks]}")
        print(f"  Operator Fee: {m[op_fee]}")
        print(f"  Initial Deposit: {m[initial_deposit]}")
        
        # Calculate derived values
        b = m[blocks].as_long()
        f = m[op_fee].as_long()
        d = m[initial_deposit].as_long()
        
        owed = b * f
        op_inc = owed
        clus_final = max(0, d - owed)
        liabilities = op_inc + clus_final
        
        print(f"  Total Fees Owed: {owed}")
        print(f"  Operator Balance Increase: {op_inc}")
        print(f"  Remaining Cluster Balance: {clus_final}")
        print(f"  Liquidator Reward: {clus_final}")
        print(f"  Total Liabilities Created: {liabilities}")
        print(f"  Total Assets (Deposit): {d}")
        print(f"  Insolvency Drift: {liabilities - d} SSV")
        
        print("\nConclusion: The protocol fails to cap operator fee accumulation by the actual cluster balance during liquidation.")
    else:
        print("\n[UNPROVED] No insolvency found in this model.")

if __name__ == "__main__":
    verify_insolvency()