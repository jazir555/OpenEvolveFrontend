# Installation and Verification Guide: SSV Formal Proofs

This document provides end-to-end instructions for installing the necessary tools and executing the formal verification suite for the SSV Network insolvency vulnerability.

---

## Part 1: Z3 Theorem Prover (Symbolic Logic)

Z3 is used to prove that the "Insolvent State" is reachable by searching for a specific combination of inputs (deposits, fees, time) that break the protocol's safety invariants.

### 1. Installation

#### Option A: Python Wrapper (Recommended for Quick Verification)
If you have Python installed, you can use the `z3-solver` library which includes the Z3 engine.
```bash
pip install z3-solver
```

#### Option B: Standalone CLI (Windows)
1. Download the latest release from [Z3 GitHub Releases](https://github.com/Z3Prover/z3/releases).
2. Extract the zip file.
3. Add the `bin` folder to your system **PATH**.
4. Verify with `z3 --version`.

#### Option C: macOS/Linux
- **macOS:** `brew install z3`
- **Linux:** `sudo apt install z3`

### 2. Executing Z3 Proofs

#### Unified Suite (Python)
We have provided a unified runner that validates all 5 Proofs of Concept:
```bash
# From the "smart contracts" directory
python run_all_z3_proofs.py
```

#### Individual SMT2 Files
To run a specific proof manually using the Z3 CLI:
```bash
z3 ssv-insolvency-poc/formal-proofs/SSV_INSOLVENCY_PROOF.smt2
```
*Note: A result of `sat` means the vulnerability is confirmed reachable.*

---

## Part 2: Lean 4 & Mathlib (Mathematical Theorems)

Lean 4 is used to prove that protocol insolvency is not just "possible" but is **mathematically guaranteed** due to the accounting mismatch.

### 1. Installation

#### Step 1: Install Elan (Lean Version Manager)
- **Windows:** Download and run `elan-init.exe` from [Elan Releases](https://github.com/leanprover/elan/releases).
- **macOS/Linux:**
  ```bash
  curl https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh -sSf | sh
  ```

#### Step 2: Verify Installation
Restart your terminal and run:
```bash
lean --version
lake --version
```

### 2. Setting Up the Proof Project

The Lean proofs require **Mathlib4** (a massive library of verified mathematics). We have already configured the `lakefile.lean` to handle this.

1. Navigate to the `smart contracts` directory.
2. Fetch and build dependencies:
   ```bash
   lake update
   ```
   *Note: This will download cached Mathlib binaries. It may take a few minutes depending on your connection.*

### 3. Compiling and Validating Proofs

To validate a Lean proof, you compile it using the Lake environment. **In Lean, a successful proof produces no output.** If it compiles without errors, the proof is verified by the Lean kernel.

#### Validate All Proofs:
```bash
lake env lean ssv-insolvency-poc/formal-proofs/ssv_global_insolvency_proof.lean
lake env lean ssv-poc2-multi-cluster/formal-proofs/multi_cluster_insolvency_proof.lean
lake env lean ssv-poc3-liquidation-griefing/formal-proofs/liquidation_griefing_proof.lean
lake env lean ssv-poc4-dao-sybil/formal-proofs/dao_insolvency.lean
lake env lean ssv-poc5-operator-sybil/formal-proofs/sybil_profitability.lean
```

---

## Summary of Files

| Tool | File Type | Purpose |
|------|-----------|---------|
| **Z3** | `.smt2` | Automated search for exploit witnesses. |
| **Lean 4** | `.lean` | Universal mathematical proof of insolvency. |
| **Python** | `.py` | Logic simulation and unified proof runners. |
| **Node.js** | `.js` | Cross-platform logic demonstration. |
| **Foundry** | `.t.sol` | Execution trace against real mainnet bytecode. |

---

## Troubleshooting

- **Z3 "Integer Expected" Error:** Ensure you are using the provided Python scripts or Z3 version 4.12.0+.
- **Lean "Unknown Module Mathlib":** Ensure you ran `lake update` inside the `smart contracts` directory.
- **Foundry RPC Error:** Ensure you have set the `MAINNET_RPC_URL` environment variable for fork-testing.
