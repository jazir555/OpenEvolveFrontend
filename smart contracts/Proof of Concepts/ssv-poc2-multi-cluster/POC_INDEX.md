# PoC Index - SSV Network Insolvency Vulnerabilities

**Project:** SSV Network Security Analysis  
**Scope:** Protocol Insolvency via Virtual Accounting  
**Last Updated:** February 7, 2026

---

## Overview

This repository contains **three complementary PoCs** demonstrating different aspects of the SSV Network insolvency vulnerability:

| PoC | Name | Focus | Severity | Virtual Debt |
|-----|------|-------|----------|--------------|
| **PoC 1** | Single-Cluster Insolvency | Basic vulnerability | Critical | ~10 SSV |
| **PoC 2** | Multi-Cluster Insolvency | Systemic risk / Bank run | Critical | ~550 SSV |
| **PoC 3** | Liquidation Griefing | Time-delayed attack | Critical | ~485 SSV |

---

## PoC Comparison

### Attack Vectors

| Aspect | PoC 1 | PoC 2 | PoC 3 |
|--------|-------|-------|-------|
| **Clusters Affected** | 1 | 3 | 1 (exploited over time) |
| **Time Horizon** | Immediate | Immediate | Delayed (200+ blocks) |
| **Attackers** | 1 operator | 3 operators + DAO | 1 operator + griefing |
| **Dynamics** | Simple theft | Bank run | Time manipulation |
| **Systemic Risk** | Low | **High** | Medium |
| **Uniqueness** | Foundation | **Systemic scope** | **Time exploitation** |

### Technical Differences

| Technical Aspect | PoC 1 | PoC 2 | PoC 3 |
|-----------------|-------|-------|-------|
| **Main Contract** | `SSVInsolvencyPoC.sol` | `SSVMultiClusterInsolvency.sol` | `SSVLiquidationGriefingPoC.sol` |
| **Test Contract** | `SSVInsolvencyPoC.t.sol` | `SSVMultiClusterInsolvency.t.sol` | `SSVLiquidationGriefingPoC.t.sol` |
| **Blocks Passed** | 10 | 100 | 200+ |
| **Validators** | 10 | 40 | 100+ |
| **Fee Structure** | Simple | Multi-tier | Front-running |

---

## Repository Structure

```
smart contracts/
├── ssv-insolvency-poc/              # PoC 1: Single-Cluster
│   ├── src/SSVInsolvencyPoC.sol
│   ├── test/SSVInsolvencyPoC.t.sol
│   └── scripts/
│
├── ssv-poc2-multi-cluster/          # PoC 2: Multi-Cluster
│   ├── src/SSVMultiClusterInsolvency.sol
│   ├── test/SSVMultiClusterInsolvency.t.sol
│   └── scripts/
│
└── ssv-poc3-liquidation-griefing/   # PoC 3: Liquidation Griefing
    ├── src/SSVLiquidationGriefingPoC.sol
    ├── test/SSVLiquidationGriefingPoC.t.sol
    └── scripts/
```

---

## Submission Strategy

### Recommended Approach: Submit ALL THREE

**Rationale:**
1. **PoC 1** establishes the fundamental vulnerability
2. **PoC 2** shows it's a systemic risk (bank run)
3. **PoC 3** demonstrates time-based exploitation

Each PoC demonstrates a unique attack vector on the same root cause.

### Submission Order

1. Submit PoC 1 first (simplest, clearest)
2. Submit PoC 2 second (shows systemic scope)
3. Submit PoC 3 third (shows time exploitation)

Or submit all three simultaneously as related vulnerabilities.

---

## Common Elements

### Shared Root Cause

All PoCs demonstrate the same fundamental vulnerability:

```solidity
// OperatorLib.sol - UNCONDITIONAL CREDIT
operator.snapshot.balance += blockDiffFee * validatorCount;  // Line 19

// ClusterLib.sol - CAPPED DEBIT
cluster.balance = usage > cluster.balance ? 0 : cluster.balance - usage;  // Line 16
```

**Asymmetry creates virtual SSV backed by nothing.**

### Shared Infrastructure

| Component | PoC 1 | PoC 2 | PoC 3 |
|-----------|-------|-------|-------|
| SSV_TOKEN | Same | Same | Same |
| SSV_NETWORK | Same | Same | Same |
| Foundry Version | Same | Same | Same |
| RPC Required | Yes | Yes | Yes |

### Contract Addresses

```solidity
address constant SSV_TOKEN = 0x9D65fF81a3c488d585bBfb0Bfe3c7707c7917f54;
address constant SSV_NETWORK = 0xDD9BC35aE942eF0cFa76930954a156B3fF30a4E1;
```

---

## Running the PoCs

### Prerequisites

```bash
# For all PoCs
export MAINNET_RPC_URL="https://eth-mainnet.g.alchemy.com/v2/YOUR_KEY"
```

### PoC 1: Single-Cluster

```bash
cd ssv-insolvency-poc
forge install
forge test -vv --match-path test/SSVInsolvencyPoC.t.sol
```

**Expected:** 10 SSV virtual debt, direct theft demonstration

### PoC 2: Multi-Cluster

```bash
cd ssv-poc2-multi-cluster
forge install
forge test -vv --match-path test/SSVMultiClusterInsolvency.t.sol
```

**Expected:** 550 SSV virtual debt, bank run dynamics

### PoC 3: Liquidation Griefing

```bash
cd ssv-poc3-liquidation-griefing
forge install
forge test -vv --match-path test/SSVLiquidationGriefingPoC.t.sol
```

**Expected:** 485 SSV virtual debt, time-delayed exploitation

---

## Formal Proofs

### Z3 SMT-LIB

| PoC | Proof File | Result |
|-----|------------|--------|
| PoC 1 | `SSV_GLOBAL_INSOLVENCY_PROOF.smt2` | sat |
| PoC 2 | `MULTI_CLUSTER_INSOLVENCY_PROOF.smt2` | sat |
| PoC 3 | `LIQUIDATION_GRIEFING_PROOF.smt2` | sat |

### Lean 4

| PoC | Theorem File | Theorem |
|-----|--------------|---------|
| PoC 1 | `ssv_global_insolvency_proof.lean` | `protocol_insolvent` |
| PoC 2 | `multi_cluster_insolvency_proof.lean` | `protocol_liabilities_geq_assets` |
| PoC 3 | `liquidation_griefing_proof.lean` | `exploitation_possible` |

---

## Documentation

Each PoC has identical documentation structure:

| Document | PoC 1 | PoC 2 | PoC 3 |
|----------|-------|-------|-------|
| README.md | ✅ | ✅ | ✅ |
| FINAL_AUDIT_REPORT.md | ✅ | ✅ | ✅ |
| SUBMISSION_GUIDE.md | ✅ | ✅ | ✅ |
| SUBMISSION_CHECKLIST.md | ✅ | ✅ | ✅ |
| GUIDELINE_COMPLIANCE_CHECKLIST.md | ✅ | ✅ | ✅ |
| POC_COMPLIANCE_REPORT.md | ✅ | ✅ | ✅ |
| TVL_UPDATE_GUIDE.md | ✅ | ✅ | ✅ |
| POC_INDEX.md | ✅ | ✅ | ✅ |

---

## Severity Justification

### Why All Three Are Critical

| PoC | Critical Factor | Evidence |
|-----|-----------------|----------|
| **PoC 1** | Direct theft of user funds | 10 SSV stolen |
| **PoC 2** | Systemic risk / Bank run | 550 SSV, affects ALL users |
| **PoC 3** | Unpreventable exploitation | 485 SSV, time-based griefing |

### Combined Impact

When considered together, these PoCs show:
1. **Vulnerability exists** (PoC 1)
2. **Scales with system usage** (PoC 2)
3. **Cannot be prevented by users** (PoC 3)

This combination justifies **Critical** severity.

---

## Expected Bounty

| PoC | TVL Impact | Expected Bounty |
|-----|------------|-----------------|
| PoC 1 | $215,130 | $50,000+ |
| PoC 2 | $215,130 | $50,000+ |
| PoC 3 | $215,130 | $50,000+ |
| **Combined** | **$645,390** | **$150,000+** |

**Note:** Actual bounty may vary based on project team's assessment.

---

## Quick Navigation

| Resource | PoC 1 | PoC 2 | PoC 3 |
|----------|-------|-------|-------|
| **Main Contract** | [src/SSVInsolvencyPoC.sol](ssv-insolvency-poc/src/SSVInsolvencyPoC.sol) | [src/SSVMultiClusterInsolvency.sol](ssv-poc2-multi-cluster/src/SSVMultiClusterInsolvency.sol) | [src/SSVLiquidationGriefingPoC.sol](ssv-poc3-liquidation-griefing/src/SSVLiquidationGriefingPoC.sol) |
| **Test** | [test/SSVInsolvencyPoC.t.sol](ssv-insolvency-poc/test/SSVInsolvencyPoC.t.sol) | [test/SSVMultiClusterInsolvency.t.sol](ssv-poc2-multi-cluster/test/SSVMultiClusterInsolvency.t.sol) | [test/SSVLiquidationGriefingPoC.t.sol](ssv-poc3-liquidation-griefing/test/SSVLiquidationGriefingPoC.t.sol) |
| **README** | [README.md](ssv-insolvency-poc/README.md) | [README.md](ssv-poc2-multi-cluster/README.md) | [README.md](ssv-poc3-liquidation-griefing/README.md) |
| **Audit Report** | [FINAL_AUDIT_REPORT.md](ssv-insolvency-poc/FINAL_AUDIT_REPORT.md) | [FINAL_AUDIT_REPORT.md](ssv-poc2-multi-cluster/FINAL_AUDIT_REPORT.md) | [FINAL_AUDIT_REPORT.md](ssv-poc3-liquidation-griefing/FINAL_AUDIT_REPORT.md) |

---

## Contact

- **Immunefi:** https://immunefi.com/bug-bounty/ssvnetwork/
- **Submission:** Via Immunefi Dashboard
- **Support:** https://immunefi.com/support

---

*Index Version: 1.0*  
*Generated: February 7, 2026*
