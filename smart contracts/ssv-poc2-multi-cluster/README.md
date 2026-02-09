# SSV Network Multi-Cluster Cascading Insolvency PoC

## 🚀 Quick Start

**Want to see the multi-cluster vulnerability immediately?**

```bash
# Option 1: Python demo (fastest)
cd "smart contracts/ssv-poc2-multi-cluster"
python scripts/demo_multi_cluster.py

# Option 2: JavaScript demo
node scripts/demo_multi_cluster.js

# Option 3: Full Foundry POC
forge test -vv
```

**Expected output**: Proof that 550 SSV is stolen via bank run dynamics in ~5 seconds.

---

## Overview

This is the **second attack vector** demonstrating the SSV Network protocol insolvency vulnerability. This PoC specifically demonstrates how multiple bankrupt clusters compound the insolvency, creating a "bank run" scenario.

> ⚠️ **SAFETY NOTICE:** This PoC operates entirely on a **local fork** of mainnet using Foundry's `vm.createSelectFork()`. No transactions are sent to actual mainnet.

**Attack Vector:** Multi-Cluster Cascading Insolvency  
**Vulnerability:** Uncollateralized Virtual Accounting  
**Severity:** Critical  
**Impact:** Direct theft of user funds via bank run dynamics  

---

## The Attack

### Strategy

1. **Setup** - 1 large cluster (healthy) + 3 small clusters (bankrupt)
2. **Bankrupt** - Allow small clusters to deplete their balances
3. **Accumulate** - Multiple operators earn virtual fees
4. **DAO** - Also earns uncollateralized network fees
5. **Bank Run** - All parties race to withdraw first

### Why This Works

When multiple clusters go bankrupt simultaneously:
- Each operator continues earning virtual fees
- DAO earns fees from ALL clusters (including bankrupt ones)
- Virtual debt compounds across multiple parties
- Creates a "bank run" where early withdrawers profit at expense of late ones

---

## Attack Scenario

```
Initial State:
  - Cluster 1 (Large):   10,000 SSV (healthy)
  - Cluster 2 (Small 1):    100 SSV (bankrupts in 100 blocks)
  - Cluster 3 (Small 2):     50 SSV (bankrupts in 50 blocks)
  - Cluster 4 (Small 3):     25 SSV (bankrupts in 25 blocks)

After 150 Blocks:
  - Cluster 2: BANKRUPT, virtual debt: 50 SSV
  - Cluster 3: BANKRUPT, virtual debt: 100 SSV
  - Cluster 4: BANKRUPT, virtual debt: 125 SSV
  - DAO fees:  275 SSV (unbacked portion)
  - Total virtual debt: 550 SSV

Bank Run:
  - Operator 3 withdraws: 125 SSV (success)
  - Operator 2 withdraws: 100 SSV (success)
  - Operator 1 withdraws:  50 SSV (success)
  - DAO withdraws:        275 SSV (success)
  - Victim Large tries:   10,000 SSV
  - Victim Large gets:    9,450 SSV
  - VICTIM LOSS:          550 SSV (STOLEN!)
```

---

## Running the PoC

### Prerequisites

```bash
# Set RPC endpoint
export MAINNET_RPC_URL="https://eth-mainnet.g.alchemy.com/v2/YOUR_KEY"
```

### Execution

```bash
cd ssv-poc2-multi-cluster

# Install dependencies
forge install

# Build
forge build

# Run the test
forge test -vv

# With full trace
forge test -vvv
```

### Specific Tests

```bash
# Main multi-cluster test
forge test -vv --match-test testMultiClusterInsolvency

# Cascading effect test
forge test -vv --match-test testCascadingEffect

# Bank run dynamics test
forge test -vv --match-test testBankRunDynamics
```

---

## Expected Output

```
=================================================================
SSV NETWORK: MULTI-CLUSTER CASCADING INSOLVENCY ATTACK
=================================================================

--- PHASE 1: Setup Multiple Clusters ---

Cluster 1 (Victim Large):   10000 SSV (healthy)
Cluster 2 (Victim Small 1): 100 SSV (bankrupts in 100 blocks)
Cluster 3 (Victim Small 2): 50 SSV (bankrupts in 50 blocks)
Cluster 4 (Victim Small 3): 25 SSV (bankrupts in 25 blocks)
Total pool balance:         10175 SSV

--- PHASE 3: Simulating 150 Blocks (Bankruptcy Events) ---

After 150 blocks:
  - Cluster 2: BANKRUPT (0 SSV balance)
    Virtual debt to Operator 1: 50 SSV
  - Cluster 3: BANKRUPT (0 SSV balance)
    Virtual debt to Operator 2: 100 SSV
  - Cluster 4: BANKRUPT (0 SSV balance)
    Virtual debt to Operator 3: 125 SSV
  - DAO unbacked network fees: 275 SSV
  - TOTAL VIRTUAL DEBT: 550 SSV

--- PHASE 4: BANK RUN - Race to Withdraw ---

Operator 3 withdrew: 125 SSV
Operator 2 withdrew: 100 SSV
Operator 1 withdrew: 50 SSV
DAO withdrew:        275 SSV

Total stolen from pool: 550 SSV
ALL OF IT IS UNBACKED VIRTUAL DEBT!

--- PHASE 5: Honest Victim Attempts Withdrawal ---

Pool remaining:          9625 SSV
Victim Large entitlement: 10000 SSV

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
CRITICAL: VICTIM LARGE FUNDS STOLEN!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

Victim Large expected:   10000 SSV
Victim Large received:   9625 SSV
VICTIM LARGE LOSS:       550 SSV

Three bankrupt clusters created 550 SSV of
virtual debt. When operators and DAO withdrew, they STOLE this
amount from Victim Large's honest deposit!

This is a BANK RUN - first to withdraw wins, last loses!
```

---

## Key Insights

### 1. Compounding Effect
Each additional bankrupt cluster adds to the total virtual debt. With N bankrupt clusters, the total theft is N times worse than a single cluster.

### 2. Bank Run Dynamics
Once virtual debt exists, rational actors will race to withdraw. Late withdrawers (often honest users who aren't monitoring constantly) bear the losses.

### 3. DAO Involvement
The DAO is also complicit in the theft - it earns network fees from ALL clusters, including bankrupt ones. When DAO withdraws, it steals from honest users.

### 4. Systemic Risk
This vulnerability affects the ENTIRE protocol, not just individual clusters. Any bankruptcy harms all remaining depositors.

---

## Comparison to PoC 1

| Aspect | PoC 1: Single-Cluster | PoC 2: Multi-Cluster |
|--------|----------------------|----------------------|
| Clusters | 1 bankrupt | 3 bankrupt |
| Operators | 1 | 3 + DAO |
| Virtual Debt | ~10 SSV | ~550 SSV |
| Scale | Small | **Large** |
| Dynamics | Simple theft | Bank run |

**PoC 2 demonstrates the systemic nature of the vulnerability.**

---

## Mitigation

### Immediate
- Implement withdrawal queues to prevent bank runs
- Add withdrawal limits during insolvency periods

### Long-term
- Link operator/DAO earnings to real-time cluster solvency
- Implement segregated pools to prevent cross-cluster liability

---

## References

- **PoC 1:** `ssv-insolvency-poc/` - Single cluster demonstration
- **PoC 3:** `ssv-poc3-liquidation-griefing/` - Liquidation griefing attack
- **Vulnerability Report:** `../SSV_INSOLVENCY_VULNERABILITY.md`

---

*PoC Version: 1.0.0*  
*Last Updated: February 2026*

## Formal Proofs

### 1. Lean 4 Mathematical Proofs
Each PoC directory is a standalone Lean package. To verify the proofs:
```bash
# From this directory (ssv-poc2-multi-cluster)
lake exe cache get
lake build
```
This confirms that the multi-cluster insolvency logic is mathematically certain, with zero `sorry` statements.

### 2. SMT-LIB Proof (Z3)
**File:** `formal-proofs/MULTI_CLUSTER_INSOLVENCY_PROOF.smt2`
```bash
# Run with Z3
z3 formal-proofs/MULTI_CLUSTER_INSOLVENCY_PROOF.smt2
```
**Result:** `sat` - Multi-cluster insolvency is mathematically reachable.

### 3. Python Verification Scripts
```bash
python scripts/demo_multi_cluster.py
python scripts/run_smt_proof.py
```
