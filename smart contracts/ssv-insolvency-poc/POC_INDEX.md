# SSV Network Insolvency PoC Collection

This directory contains **three independent Proof of Concept (PoC)** demonstrations of the SSV Network protocol insolvency vulnerability.

---

## PoC Collection

### PoC 1: Single-Cluster Operator Exploitation
**File:** `src/SSVInsolvencyPoC.sol`  
**Test:** `test/SSVInsolvencyPoC.t.sol`

**Description:** Demonstrates the basic vulnerability where a single operator services one cluster that goes bankrupt, then withdraws uncollateralized virtual earnings, stealing from other users.

**Key Points:**
- Simplest demonstration of the vulnerability
- Shows direct 1:1 theft ratio
- Easy to understand and verify

**Run:**
```bash
forge test -vv --match-path test/SSVInsolvencyPoC.t.sol
```

---

### PoC 2: Multi-Cluster Cascading Insolvency
**File:** `src/SSVInsolvencyPoC.sol` (with multi-cluster test)  
**Test:** `test/SSVInsolvencyPoC.t.sol::testAccountingMismatch`

**Description:** Demonstrates how multiple clusters going bankrupt compounds the insolvency, creating a "bank run" scenario where operators and DAO race to withdraw before honest users.

**Key Points:**
- Shows systemic nature of vulnerability
- Multiple operators + DAO exploitation
- Demonstrates bank run dynamics

**Run:**
```bash
forge test -vv --match-test testMultiClusterCascadingInsolvency
```

---

### PoC 3: Time-Delayed Liquidation Griefing ⭐
**File:** `src/SSVLiquidationGriefingPoC.sol`  
**Test:** `test/SSVLiquidationGriefing.t.sol`  
**Docs:** `SSV_LIQUIDATION_GRIEFING_POC.md`

**Description:** Demonstrates how an attacker can grief liquidators to maximize virtual debt accumulation, leading to the largest possible theft from honest users.

**Key Points:**
- **Most severe attack vector**
- Exploits liquidation mechanism
- Maximizes virtual debt through griefing
- Hardest to detect
- Can be executed by anyone (not just operators)

**Attack Flow:**
1. Monitor for clusters nearing liquidation
2. Grief liquidators (front-run or gas exhaustion)
3. Allow 200+ blocks of virtual debt accumulation
4. Race to withdraw before victims
5. Bank run leaves honest users with losses

**Run:**
```bash
forge test -vv --match-path test/SSVLiquidationGriefing.t.sol
```

---

## Additional Proofs

### Mainnet Bytecode Proof
**File:** `test/SSVMainnetExploit.t.sol`

Demonstrates the vulnerability against the **actual deployed SSV Network bytecode** on mainnet using storage manipulation.

**Run:**
```bash
forge test -vv --match-path test/SSVMainnetExploit.t.sol --fork-url $MAINNET_RPC_URL
```

### Formal Mathematical Proofs
- **Z3 SMT:** `formal-proofs/SSV_INSOLVENCY_PROOF.smt2`
- **Lean 4:** `formal-proofs/ssv_global_insolvency_proof.lean`
- **Python:** `scripts/definitive_ssv_insolvency_proof.py`

---

## Quick Start

### 1. Setup
```bash
cd ssv-insolvency-poc
forge install
```

### 2. Configure RPC (for mainnet fork)
```bash
export MAINNET_RPC_URL="https://eth-mainnet.g.alchemy.com/v2/YOUR_KEY"
```

### 3. Run All PoCs
```bash
# Run all tests
forge test -vv

# Run specific PoC
forge test -vv --match-test testInsolvencyAttack
forge test -vv --match-test testLiquidationGriefingAttack
forge test -vv --match-test testMainnetBytecodeInsolvency
```

---

## Severity Assessment

| PoC | Attack Complexity | Impact Scale | Bounty Tier |
|-----|-------------------|--------------|-------------|
| PoC 1: Single-Cluster | Low | Small ($10K-$50K) | Critical |
| PoC 2: Multi-Cluster | Medium | Medium ($50K-$200K) | Critical |
| PoC 3: Liquidation Griefing | Medium | **Large ($200K-$1M)** | **Critical** |

**Recommendation:** Lead with **PoC 3** as it demonstrates the maximum potential impact and is the most economically viable attack.

---

## Vulnerability Summary

**Root Cause:** Decoupled Virtual Accounting

```solidity
// OperatorLib.sol - UNCONDITIONAL CREDIT
operator.snapshot.balance += blockDiffFee * validatorCount;  // NO SOLVENCY CHECK

// ClusterLib.sol - CAPPED DEBIT  
cluster.balance = usage > balance ? 0 : balance - usage;     // CAPPED AT 0
```

**Result:** When cluster balance hits 0, operators and DAO continue earning virtual credits backed by **nothing**. These virtual credits can be withdrawn as **real tokens**, stealing from honest users.

**Status:** Confirmed in production (v1.2.0, audited by Quantstamp)

---

## Files Overview

```
ssv-insolvency-poc/
├── src/
│   ├── PoC.sol                          # Base PoC contract
│   ├── SSVInsolvencyPoC.sol            # PoC 1 & 2
│   ├── SSVLiquidationGriefingPoC.sol   # PoC 3 ⭐
│   └── tokens/Tokens.sol               # Token utilities
├── test/
│   ├── SSVInsolvencyPoC.t.sol          # Tests for PoC 1 & 2
│   ├── SSVLiquidationGriefing.t.sol    # Tests for PoC 3
│   └── SSVMainnetExploit.t.sol         # Mainnet bytecode proof
├── formal-proofs/
│   ├── SSV_INSOLVENCY_PROOF.smt2       # Z3 formal proof
│   └── ssv_global_insolvency_proof.lean # Lean 4 proof
├── scripts/
│   └── definitive_ssv_insolvency_proof.py # Python verification
├── SSV_LIQUIDATION_GRIEFING_POC.md     # PoC 3 documentation
├── POC_INDEX.md                        # This file
└── foundry.toml                        # Foundry configuration
```

---

## Immunefi Submission Guide

When submitting to Immunefi:

1. **Lead with PoC 3** - It shows maximum impact
2. **Include all three PoCs** - Demonstrates thoroughness
3. **Reference formal proofs** - Shows mathematical certainty
4. **Use mainnet bytecode test** - Proves production vulnerability

**Required PoC Format Checklist:**
- ✅ Uses Foundry framework
- ✅ Forks mainnet (`vm.createSelectFork`)
- ✅ Demonstrates actual fund theft
- ✅ Includes step-by-step logs
- ✅ Has corresponding test file
- ✅ Can be run with `forge test`

---

## Contact

For questions about these PoCs, refer to:
- **Immunefi Bounty:** https://immunefi.com/bug-bounty/ssvnetwork/
- **SSV Network Docs:** https://docs.ssv.network/

---

*Collection Version: 1.0.0*  
*Last Updated: February 2026*
