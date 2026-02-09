# SSV Network Insolvency Vulnerability - PoC Collection

This directory contains **five independent Proof of Concept (PoC)** demonstrations of the SSV Network protocol insolvency vulnerability, each with:
- Solidity exploits
- Foundry tests
- Python scripts
- **JavaScript/Hardhat scripts** (for major vectors)
- Formal mathematical proofs

---

## 📁 Complete Folder Structure

```
smart contracts/
├── ssv-insolvency-poc/                    # PoC 1: Original/Base
│   ├── src/
│   ├── test/
│   ├── scripts/                           # Python + JavaScript
│   ├── formal-proofs/
│   ├── foundry.toml
│   └── README.md
│
├── ssv-poc2-multi-cluster/                # PoC 2: Multi-Cluster ⭐
│   ├── src/
│   ├── test/
│   ├── scripts/                           # Python + JavaScript
│   ├── formal-proofs/
│   ├── foundry.toml
│   └── README.md
│
├── ssv-poc3-liquidation-griefing/         # PoC 3: Liquidation Griefing ⭐⭐
│   ├── src/
│   ├── test/
│   ├── scripts/                           # Python + JavaScript
│   ├── formal-proofs/
│   ├── foundry.toml
│   └── README.md
│
├── ssv-poc4-dao-sybil/                    # PoC 4: DAO Sybil Inflation ⭐
│   ├── src/
│   ├── test/
│   ├── scripts/                           # Python + JavaScript
│   ├── formal-proofs/
│   ├── foundry.toml
│   └── README.md
│
├── ssv-poc5-operator-sybil/                 # PoC 5: Operator Sybil Self-Dealing ⭐
│   ├── src/
│   ├── test/
│   ├── scripts/                           # Python + JavaScript
│   ├── formal-proofs/
│   ├── foundry.toml
│   └── README.md
│
└── POC_COLLECTION_INDEX.md                # This file
```

---

## 🚀 Quick Start (All Languages)

### Solidity/Foundry
```bash
# PoC 1
cd ssv-insolvency-poc && forge test -vv

# PoC 2
cd ssv-poc2-multi-cluster && forge test -vv

# PoC 3
cd ssv-poc3-liquidation-griefing && forge test -vv

# PoC 4
cd ssv-poc4-dao-sybil && forge test -vv

# PoC 5
cd ssv-poc5-operator-sybil && forge test -vv
```

### Lean 4 (Mathematical Proofs)
Each POC folder contains its own Lean environment. To verify the proofs:
```bash
# Example for PoC 1 (Repeat for other folders)
cd ssv-insolvency-poc
lake build
```
This will download dependencies (Mathlib) and compile the proofs, confirming zero `sorry` statements.

### Python
```bash
# Run all Z3 proofs
python run_all_z3_proofs.py

# Individual Python Demos
python ssv-insolvency-poc/scripts/run_execution_poc.py
python ssv-poc2-multi-cluster/scripts/demo_multi_cluster.py
python ssv-poc3-liquidation-griefing/scripts/demo_griefing.py
python ssv-poc4-dao-sybil/scripts/demo_dao_sybil.py
python ssv-poc5-operator-sybil/scripts/demo_operator_sybil.py
```

---

## 📊 Complete Verification Matrix

| Verification Method | PoC 1 | PoC 2 | PoC 3 | PoC 4 | PoC 5 |
|---------------------|-------|-------|-------|-------|-------|
| **Solidity PoC** | ✅ | ✅ | ✅ | ✅ | ✅ |
| **Foundry Test** | ✅ | ✅ | ✅ | ✅ | ✅ |
| **Python Demo** | ✅ | ✅ | ✅ | ✅ | ✅ |
| **Python Z3 Verify** | ✅ | ✅ | ✅ | ✅ | ✅ |
| **JavaScript Demo** | ✅ | ✅ | ✅ | ✅ | ✅ |
| **Z3 SMT-LIB** | ✅ | ✅ | ✅ | ✅ | ✅ |
| **Lean 4 Proof** | ✅ | ✅ | ✅ | ✅ | ✅ |
| **Mainnet Fork** | ✅ | ✅ | ✅ | ✅ | ✅ |

---

## 🎯 PoC Overview

### PoC 1: Single-Cluster Operator Exploitation
**Location:** `ssv-insolvency-poc/`
**Description:** Basic vulnerability where a single operator services one bankrupt cluster.

### PoC 2: Multi-Cluster Cascading Insolvency ⭐
**Location:** `ssv-poc2-multi-cluster/`
**Description:** Multiple clusters going bankrupt compound insolvency.

### PoC 3: Time-Delayed Liquidation Griefing ⭐⭐ (MOST SEVERE)
**Location:** `ssv-poc3-liquidation-griefing/`
**Description:** Attacker griefs liquidators to maximize virtual debt.

### PoC 4: DAO Sybil Fee Inflation ⭐
**Location:** `ssv-poc4-dao-sybil/`
**Description:** DAO treasury earns unbacked fees from spammed "dust" clusters.

### PoC 5: Operator Sybil Self-Dealing ⭐
**Location:** `ssv-poc5-operator-sybil/`
**Description:** Malicious operator creates bankrupt minions to extract infinite yield.

---

## 📝 Vulnerability Summary

**Root Cause:** Decoupled Virtual Accounting

```solidity
// OperatorLib.sol - UNCONDITIONAL CREDIT
operator.snapshot.balance += blockDiffFee * validatorCount;  // NO SOLVENCY CHECK

// ClusterLib.sol - CAPPED DEBIT  
cluster.balance = usage > balance ? 0 : balance - usage;     // CAPPED AT 0
```

**Impact:** Direct theft of user funds  
**Severity:** CRITICAL  
**Status:** Confirmed in production (v1.2.0)  
**Bounty:** Up to $1,000,000

---

## 🔗 External References

- **Immunefi Bounty:** https://immunefi.com/bug-bounty/ssvnetwork/
- **SSV Network Docs:** https://docs.ssv.network/
- **SSV Token:** 0x9D65fF81a3c488d585bBfb0Bfe3c7707c7917f54
- **SSV Network:** 0xDD9BC35aE942eF0cFa76930954a156B3fF30a4E1

---

## ✅ Complete Checklist

- [x] PoC 1: Single-cluster demonstration
- [x] PoC 2: Multi-cluster cascading attack  
- [x] PoC 3: Liquidation griefing attack
- [x] All PoCs in separate folders
- [x] Solidity exploit contracts
- [x] Foundry test files
- [x] **Python execution scripts**
- [x] **Python Z3 verification scripts**
- [x] **JavaScript execution scripts** ⭐
- [x] **JavaScript verification scripts** ⭐
- [x] **Hardhat test scripts** ⭐
- [x] **Advanced griefing techniques (PoC 3)** ⭐
- [x] Z3 SMT-LIB formal proofs
- [x] Lean 4 mathematical proofs
- [x] Foundry.toml configurations
- [x] Comprehensive README files

---

*Collection Version: 1.1.0*  
*Last Updated: February 2026*
