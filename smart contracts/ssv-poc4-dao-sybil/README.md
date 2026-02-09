# SSV DAO Sybil Fee Inflation PoC (POC 4)

## 🚀 Quick Start

**Want to see the DAO Sybil attack immediately?**

```bash
# Option 1: Python demo (fastest)
cd "smart contracts/ssv-poc4-dao-sybil"
python scripts/demo_dao_sybil.py

# Option 2: JavaScript demo
node scripts/demo_dao_sybil.js

# Option 3: Full Foundry POC
forge test -vv
```

**Expected output**: Proof that DAO Sybil attack can steal 12,000 SSV in ~5 seconds.

---

## Overview

This is the **fourth attack vector** demonstrating the SSV Network protocol insolvency vulnerability. 

**Angle:** "DAO Fee Inflation via Dust Clusters"
**Attacker:** Any user (Operator status NOT required for the core setup, though helps).
**Mechanism:** Sybil Attack + DAO Global Fee Logic.

While previous POCs focused on Operator theft, this POC demonstrates that the **DAO itself** becomes the vehicle for insolvency. By spamming "Dust Clusters" (small deposits) and letting them rot, an attacker forces the protocol to mint unbacked SSV to the DAO treasury. When the DAO (or anyone through shared governance) withdraws these fees, they drain the backing of honest users.

## The Attack

1.  **Sybil Setup:** Attacker creates 50 "Dust Clusters" with minimal SSV.
2.  **Bankruptcy:** These clusters burn their fuel in ~20 blocks.
3.  **Zombie State:** The clusters remain active in the protocol's eyes for fee calculation.
4.  **Inflation:** The DAO earns fees from ALL clusters unconditionally.
    *   `DAO_Earnings = Global_Index * Validators`
5.  **Extraction:** The DAO withdraws its "earned" fees. Since the clusters were bankrupt, these fees are printed out of thin air, stealing user principal.

## Running the PoC

```bash
cd ssv-poc4-dao-sybil
forge test -vv
```

## Formal Proofs

This POC includes:
1.  **Z3 Proof (`formal-proofs/DAO_INSOLVENCY.smt2`):** Proves that `DAO_Liabilities` can grow unbounded while `Total_Assets` remain constant.
2.  **Lean 4 Mathematical Proofs**
    Each PoC directory is a standalone Lean package. To verify the proofs:
    ```bash
    # From this directory (ssv-poc4-dao-sybil)
    lake exe cache get
    lake build
    ```
    Theorem proving the divergence of DAO claims vs actual collateral.

