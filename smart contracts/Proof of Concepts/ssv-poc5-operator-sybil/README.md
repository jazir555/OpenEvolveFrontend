# SSV Operator Sybil "Self-Dealing" PoC (POC 5)

## 🚀 Quick Start

**Want to see the Operator Sybil attack immediately?**

```bash
# Option 1: Python demo (fastest)
cd "smart contracts/ssv-poc5-operator-sybil"
python scripts/demo_operator_sybil.py

# Option 2: JavaScript demo
node scripts/demo_operator_sybil.js

# Option 3: Full Foundry POC
forge test -vv
```

**Expected output**: Proof that 250 SSV investment generates 9,750 SSV revenue (3,900% ROI) in ~5 seconds.

---

## Overview

This is the **fifth and final attack vector** demonstrating the SSV Network protocol insolvency vulnerability.

**Angle:** "Industrial Scale Self-Dealing"
**Attacker:** Malicious Operator.
**Mechanism:** Sybil Attack (Self-Delegation).

This attack demonstrates the massive scalability of the vulnerability. An operator does not need to wait for victims to go bankrupt. The operator can **create** the bankrupt victims themselves using Sybil accounts.

## The Attack

1.  **Investment:** Attacker invests a small amount (e.g., 250 SSV) to fund 50 "Minion" validators.
2.  **Self-Delegation:** Minions delegate to the Attacker's Operator.
3.  **Bankruptcy:** Minions go bankrupt almost immediately.
4.  **Infinite Yield:** The Attacker's Operator continues to accrue fees from 50 sources simultaneously, despite them having no funds.
5.  **Profit:** The Attacker withdraws the "virtual" earnings, which effectively wash trade the initial investment into a claim 40x larger, stolen from honest users.

## Running the PoC

```bash
cd ssv-poc5-operator-sybil
forge test -vv
```

## Formal Proofs

This POC includes:
1.  **Z3 Proof (`formal-proofs/OPERATOR_PROFIT.smt2`):** Proves that `Operator_Profit` > `Initial_Investment` is always satisfiable given `t > bankruptcy_threshold`.
2.  **Lean 4 Mathematical Proofs**
    Each PoC directory is a standalone Lean package. To verify the proofs:
    ```bash
    # From this directory (ssv-poc5-operator-sybil)
    lake exe cache get
    lake build
    ```
    Theorem proving infinite ROI potential.

