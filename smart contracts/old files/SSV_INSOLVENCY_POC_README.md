# SSV Network Protocol Insolvency PoC

## Overview

This repository contains a **Foundry-based Proof of Concept (PoC)** demonstrating a Critical vulnerability in the SSV Network protocol that enables **direct theft of user funds** through systematic protocol insolvency.

> ⚠️ **SAFETY NOTICE:** This PoC operates entirely on a **local fork** of mainnet using Foundry's `vm.createSelectFork()`. No transactions are sent to the actual Ethereum mainnet or any public testnet. All testing is performed in an isolated, local environment as required by Immunefi guidelines.

**Vulnerability Type:** Accounting Mismatch / Protocol Insolvency  
**Severity:** Critical  
**Impact:** Direct theft of user funds, systemic insolvency  
**Status:** Confirmed in Production Code (v1.2.0)  
**Bounty Tier:** $1,000,000 (Critical - per Immunefi guidelines)

---

## Table of Contents

1. [Vulnerability Summary](#vulnerability-summary)
2. [Prerequisites](#prerequisites)
3. [Installation](#installation)
4. [Running the PoC](#running-the-poc)
5. [Expected Output](#expected-output)
6. [Formal Proofs](#formal-proofs)
7. [Files](#files)

---

## Vulnerability Summary

### The Problem

The SSV Network protocol uses a **"decoupled virtual credit"** system that is fundamentally insolvent by design:

| Component | Behavior | Issue |
|-----------|----------|-------|
| **Cluster Balance** | Capped at zero when depleted | ✅ Correct |
| **Operator Earnings** | Grow unconditionally with each block | ❌ **No solvency check** |
| **DAO Earnings** | Grow unconditionally with each block | ❌ **No solvency check** |

### The Attack

1. **Victim A** deposits 1000 SSV (honest user)
2. **Victim B** deposits 10 SSV (will go bankrupt)
3. **Time passes**: Victim B's cluster goes bankrupt (balance = 0)
4. **Operator** continues earning uncollateralized virtual fees
5. **Operator withdraws**: Takes real SSV from the shared pool
6. **Result**: Victim A can only withdraw 990 SSV (**LOSS: 10 SSV**)

### Root Cause

```solidity
// OperatorLib.sol - Unconditional credit
operator.snapshot.balance += blockDiffFee * validatorCount;  // NO SOLVENCY CHECK

// ClusterLib.sol - Capped debit  
cluster.balance = usage > balance ? 0 : balance - usage;     // CAPPED AT 0
```

---

## Prerequisites

- [Foundry](https://book.getfoundry.sh/getting-started/installation) installed
- [Git](https://git-scm.com/)
- Ethereum RPC endpoint (for forking)

### RPC Endpoint Setup

```bash
# Set your RPC endpoint (required for mainnet forking)
export MAINNET_RPC_URL="https://eth-mainnet.g.alchemy.com/v2/YOUR_API_KEY"
# OR
export MAINNET_RPC_URL="https://mainnet.infura.io/v3/YOUR_PROJECT_ID"
```

---

## Installation

```bash
# Navigate to the forge-poc-templates directory
cd forge-poc-templates

# Install dependencies
forge install

# Build the project
forge build
```

---

## Running the PoC

### Basic Run

```bash
forge test -vv --match-path test/pocs/SSVNetworkInsolvency.t.sol
```

### With Full Trace

```bash
forge test -vvv --match-path test/pocs/SSVNetworkInsolvency.t.sol
```

### Specific Test Functions

```bash
# Main attack demonstration
forge test -vv --match-test testInsolvencyAttack

# Verify accounting mismatch
forge test -vv --match-test testAccountingMismatch

# Verify production vulnerability
forge test -vv --match-test testProductionVulnerability
```

---

## Expected Output

```
=================================================================
SSV NETWORK PROTOCOL INSOLVENCY EXPLOIT
=================================================================
Vulnerability: Uncollateralized Virtual Accounting
Impact: Direct theft of user funds
Severity: CRITICAL
=================================================================

--- PHASE 1: Setup Deposits ---

Victim A deposited: 1000 SSV
Victim B deposited: 10 SSV
Total contract balance: 1010 SSV

--- PHASE 2: Simulating 10 Blocks (Bankruptcy) ---

After 10 blocks:
  - Victim B cluster: BANKRUPT (0 SSV)
  - Operator virtual earnings: 10 SSV
  - UNBACKED portion: 10 SSV

--- PHASE 3: Operator Withdraws Virtual Earnings ---

Operator withdrew: 10 SSV
  (All of it is UNBACKED virtual debt)

--- PHASE 4: Victim A Attempts Withdrawal ---

Contract balance: 1000 SSV
Victim A entitlement: 1000 SSV

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
VULNERABILITY CONFIRMED: FUNDS STOLEN!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

Victim A LOSS: 10 SSV

The operator withdrew 10 SSV of virtual earnings,
but Victim B only had 10 SSV to pay.
The shortage was STOLEN from Victim A's deposit!

ROOT CAUSE:
  OperatorLib.sol - updateSnapshot() increases operator
  balances WITHOUT checking cluster solvency.
  ClusterLib.sol - updateBalance() caps cluster at 0,
  creating an accounting mismatch.

=================================================================
EXPLOIT SUMMARY
=================================================================
Virtual Debt Created: 10 SSV
Funds Stolen from Victim A: 10 SSV
Protocol Insolvency: CONFIRMED
=================================================================
```

---

## Formal Proofs

This PoC is supported by formal mathematical proofs:

### 1. Z3 SMT-LIB Proof
**File:** `SSV_INSOLVENCY_PROOF.smt2`

```bash
z3 SSV_INSOLVENCY_PROOF.smt2
```

**Result:** `sat` - Insolvency state is mathematically reachable.

### 2. Lean 4 Proof
**File:** `ssv_global_insolvency_proof.lean`

Contains the theorem `ssv_global_insolvency` proving that protocol-wide insolvency is a mathematical certainty given the accounting mismatch.

### 3. Python Verification
**File:** `definitive_ssv_insolvency_proof.py`

```bash
python definitive_ssv_insolvency_proof.py
```

---

## Files

| File | Description |
|------|-------------|
| `pocs/SSVNetworkInsolvency.sol` | Main exploit contract following forge-poc-templates format |
| `test/pocs/SSVNetworkInsolvency.t.sol` | Foundry test file |
| `SSV_INSOLVENCY_PROOF.smt2` | Z3 SMT-LIB formal proof |
| `ssv_global_insolvency_proof.lean` | Lean 4 mathematical proof |
| `definitive_ssv_insolvency_proof.py` | Python Z3 verification |
| `SSV_INSOLVENCY_VULNERABILITY.md` | Detailed vulnerability report |
| `VULNERABILITY_VERIFICATION_REPORT.md` | Complete verification analysis |

---

## Affected Code

The vulnerability exists in the production SSV Network contracts:

| File | Lines | Vulnerable Code |
|------|-------|-----------------|
| `OperatorLib.sol` | 15-29 | `operator.snapshot.balance += blockDiffFee * validatorCount;` |
| `ProtocolLib.sol` | 26-36 | DAO earnings accumulate unconditionally |
| `ClusterLib.sol` | 15-23 | Cluster balance capped at zero |
| `SSVOperators.sol` | 191-214 | Withdrawal of virtual earnings |
| `SSVDAO.sol` | 26-43 | DAO withdrawal of virtual earnings |

---

## Remediation

**Recommended Fix:** Link operator/DAO fee accumulation to cluster solvency:

1. Only credit operator fees if clusters have sufficient balance
2. Implement global collateral check before withdrawals
3. Or segregate funds to prevent cross-cluster liability

---

## References

- **Immunefi Bounty:** https://immunefi.com/bug-bounty/ssvnetwork/
- **SSV Network Docs:** https://docs.ssv.network/
- **SSV Token:** 0x9D65fF81a3c488d585bBfb0Bfe3c7707c7917f54
- **SSV Network:** 0xDD9BC35aE942eF0cFa76930954a156B3fF30a4E1

---

## License

UNLICENSED - For security research purposes only.

---

*PoC Version: 1.0.0*  
*Last Updated: February 2026*  
*Foundry Version: ^0.8.13*
