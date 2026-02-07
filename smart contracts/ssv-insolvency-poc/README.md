# PoC: Systematic Protocol Insolvency in ssv.network

## Overview

This repository contains a **Foundry-based Proof of Concept (PoC)** demonstrating a Critical vulnerability in the ssv.network protocol that enables **direct theft of user funds** through systematic protocol insolvency.

> ⚠️ **SAFETY NOTICE:** This PoC operates entirely on a **local fork** of mainnet 
> using Foundry's `vm.createSelectFork()`. No transactions are sent to the actual 
> Ethereum mainnet or any public testnet. All testing is performed in an isolated, 
> local environment as required by Immunefi guidelines. This PoC does NOT perform 
> any DoS attacks.
>
> **Important:** This PoC forks mainnet to test against the actual deployed 
> SSV Network contracts, ensuring accurate test conditions as required by 
> Immunefi guidelines.

**Vulnerability Type:** Accounting Mismatch / Protocol Insolvency  
**Severity:** Critical  
**Impact:** Direct theft of user funds, systemic insolvency  
**Status:** Confirmed in Production Code (v1.2.0)

---

## Table of Contents

1. [Vulnerability Summary](#vulnerability-summary)
2. [Prerequisites](#prerequisites)
3. [Installation](#installation)
4. [Running the PoC](#running-the-poc)
5. [Expected Output](#expected-output)
6. [Formal Proofs](#formal-proofs)
7. [Funds at Risk](#funds-at-risk)

---

## Vulnerability Summary

### The Problem

The ssv.network protocol uses a **"decoupled virtual credit"** system that is fundamentally insolvent by design:

| Component | Behavior | Issue |
|-----------|----------|-------|
| **Cluster Balance** | Capped at zero when depleted | ✅ Correct |
| **Operator Earnings** | Grow unconditionally with each block | ❌ **No solvency check** |
| **DAO Earnings** | Grow unconditionally with each block | ❌ **No solvency check** |

### The Attack

1. **User A** deposits 1000 SSV (honest user)
2. **User B** deposits 10 SSV (will go bankrupt)
3. **Time passes**: User B's cluster goes bankrupt (balance = 0)
4. **Operator** continues earning uncollateralized virtual fees
5. **Operator withdraws**: Takes real SSV from the shared pool
6. **Result**: User A can only withdraw 960 SSV (**LOSS: 40 SSV**)

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
- Git

---

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd ssv-insolvency-poc

# Install dependencies
forge install

# Build the project
forge build
```

---

## Running the PoC

### Prerequisites for Testing

This PoC uses **mainnet forking** to test against actual deployed SSV Network contracts (per Immunefi guidelines). You need an Ethereum RPC endpoint.

```bash
# Set your RPC endpoint (required)
export MAINNET_RPC_URL="https://eth-mainnet.g.alchemy.com/v2/YOUR_API_KEY"
# OR
export MAINNET_RPC_URL="https://mainnet.infura.io/v3/YOUR_PROJECT_ID"
```

### Basic Run

```bash
forge test -vv --match-path test/SSVInsolvencyPoC.t.sol
```

### With Full Trace

```bash
forge test -vvv --match-path test/SSVInsolvencyPoC.t.sol
```

### Specific Test Functions

```bash
# Main attack demonstration (demonstrates full vulnerability)
forge test -vv --match-test testInsolvencyAttack

# Verify accounting mismatch
forge test -vv --match-test testAccountingMismatch
```

---

## Expected Output

```
>>> SSV Network Protocol Insolvency Demonstration
>>> Vulnerability: Uncollateralized Virtual Accounting

>>> Step 1: Initial Deposits
User A deposits: 1000 SSV
User B deposits: 10 SSV
Total contract assets: 1010 SSV

>>> Step 2: Time Passes (10 blocks)
Operator fee: 5 SSV/block
User B cluster burn rate: 5 SSV/block
User B cluster balance after 10 blocks: 0 SSV (BANKRUPT)
Operator virtual balance: 50 SSV (UNCOLLATERALIZED)

>>> Step 3: Operator Withdraws Virtual Earnings
Operator withdraws: 50 SSV
Contract balance after withdrawal: 960 SSV

>>> Step 4: Honest User A Attempts Withdrawal
User A is entitled to: 1000 SSV
Contract has: 960 SSV
CRITICAL: User A can only withdraw: 960 SSV
USER A LOSS: 40 SSV
These funds were stolen to pay uncollateralized operator debt!

>>> VULNERABILITY CONFIRMED
Protocol deficit: 40 SSV
```

---

## Formal Proofs

This PoC includes multiple formal verification methods:

### 1. SMT-LIB Proof (Z3)
**File:** `formal-proofs/SSV_INSOLVENCY_PROOF.smt2`

```bash
# Run with Z3
z3 formal-proofs/SSV_INSOLVENCY_PROOF.smt2
```

**Result:** `sat` - Insolvency state is mathematically reachable.

### 2. Lean 4 Proof
**Files:** 
- `formal-proofs/ssv_insolvency_mathlib_proof.lean`
- `formal-proofs/ssv_global_insolvency_proof.lean`

These files contain formal theorems proving that protocol insolvency is a mathematical certainty given the accounting mismatch.

### 3. Python Verification Scripts
**Files:**
- `scripts/run_execution_poc.py` - Execution trace simulation
- `scripts/verify_ssv_global_insolvency.py` - Z3-based proof

```bash
python scripts/run_execution_poc.py
python scripts/verify_ssv_global_insolvency.py
```

---

## Funds at Risk

**Vault Address:** `0x2Be7549f1B58Fc3E81427a09E61e6D0B050A4C1D`  
**Data Source:** Immunefi Bounty Program / Etherscan  
**Last Updated:** February 2026

| Metric | Value |
|--------|-------|
| Total Value Locked (TVL) | ~60,600 SSV |
| Funds Available in Vault | $215,176.19 USD |
| 30d Avg Funds Availability | $245,765.56 USD |
| Average Price of SSV | ~$3.55 USD |
| **Total Funds at Risk** | **~$215,130 USD** |

### Bounty Calculation

Per Immunefi's Critical severity formula (10% of funds at risk, min $50,000):
- 10% of $215,130 = $21,513
- **Minimum Bounty: $50,000 USD** (applies)
- Maximum Bounty: $1,000,000 USD

---

## Project Structure

```
ssv-insolvency-poc/
├── foundry.toml              # Foundry configuration
├── README.md                 # This file
├── src/
│   ├── PoC.sol              # Immunefi PoC base contract
│   ├── SSVInsolvencyPoC.sol # Main attack contract
│   ├── log/                 # Logging utilities
│   └── tokens/              # Token utilities
├── test/
│   └── SSVInsolvencyPoC.t.sol # Foundry test file
├── scripts/
│   ├── run_execution_poc.py          # Python execution trace
│   ├── run_smt_proof.py              # SMT proof runner
│   └── verify_ssv_global_insolvency.py # Z3 verification
└── formal-proofs/
    ├── SSV_INSOLVENCY_PROOF.smt2          # Z3 SMT-LIB proof
    ├── ssv_insolvency_mathlib_proof.lean  # Lean 4 proof
    ├── ssv_global_insolvency_proof.lean   # Lean 4 theorem
    └── SSV_FORMAL_PROOF_CERTIFICATE.json  # Proof certificate
```

---

## Affected Code

The vulnerability exists in the production ssv.network contracts:

| File | Lines | Vulnerable Code |
|------|-------|-----------------|
| `OperatorLib.sol` | 15-28 | `operator.snapshot.balance += blockDiffFee * validatorCount;` |
| `ProtocolLib.sol` | 26-36 | DAO earnings accumulate unconditionally |
| `ClusterLib.sol` | 15-22 | Cluster balance capped at zero |

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
- **Vault Address:** 0x2Be7549f1B58Fc3E81427a09E61e6D0B050A4C1D

---

## License

UNLICENSED - For security research purposes only.

---

*PoC Version: 1.0.0*  
*Last Updated: February 2026*  
*Foundry Version: ^0.8.13*
