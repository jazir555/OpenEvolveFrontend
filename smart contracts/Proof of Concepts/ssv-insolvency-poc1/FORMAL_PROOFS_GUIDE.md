# Guide to Formal Proofs: SSV Protocol Insolvency

This document explains the formal verification artifacts included in this PoC. These proofs provide mathematical certainty that the ssv.network accounting logic is fundamentally insolvent, moving beyond simple simulation to rigorous logical verification.

---

## 0. Environment Setup

To verify these proofs, you will need to install the Z3 Theorem Prover and the Lean 4 Theorem Prover.

### 1. Installing Z3 Theorem Prover
Z3 is a state-of-the-art SMT solver from Microsoft Research.

*   **GitHub:** [https://github.com/Z3Prover/z3](https://github.com/Z3Prover/z3)
*   **Installation:**
    *   **Windows:** 
        *   Download the latest release zip (e.g., `z3-x.x.x-x64-win.zip`) from [Z3 Releases](https://github.com/Z3Prover/z3/releases).
        *   Extract the zip and add the `bin` folder to your system **PATH**.
        *   Alternatively, using Chocolatey: `choco install z3`
    *   **macOS:** `brew install z3`
    *   **Linux (Ubuntu/Debian):** `sudo apt install z3`
*   **Verification:** Run `z3 --version` in your terminal.

### 2. Installing Lean 4
Lean 4 is installed via `elan`, the Lean version manager.

*   **Official Guide:** [https://leanprover-community.github.io/get_started.html](https://leanprover-community.github.io/get_started.html)
*   **Installation:**
    *   **Unix (macOS/Linux):**
        ```bash
        curl https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh -sSf | sh
        ```
    *   **Windows:**
        *   Download and run the `elan-init.exe` from [elan releases](https://github.com/leanprover/elan/releases).
        *   Follow the on-screen instructions to set up the default toolchain (usually `leanprover/lean4:stable`).
*   **Verification:** Run `lean --version` and `lake --version` in your terminal.
*   **Note:** These proofs require `Mathlib4`. The provided PoC directory structure is designed to be recognized by `lake` if you initialize it as a Lean project.

---

## 1. Z3 SMT-LIB Reachability Proof
**File:** `formal-proofs/SSV_INSOLVENCY_PROOF.smt2`

### What it is
This is a symbolic logic model written in the standard SMT-LIB v2.6 language. It defines the protocol's accounting rules as a set of mathematical constraints.

### What it proves
It proves that an **Insolvent State is Reachable**. Specifically, it asks the Z3 solver: *"Is there any combination of deposits, fees, and time where the tokens the protocol promises to pay (Liabilities) exceed the tokens it actually holds (Assets)?"*

### How to execute
You can run this using the Z3 Prover (available on most platforms):
```bash
z3 formal-proofs/SSV_INSOLVENCY_PROOF.smt2
```

### How to verify the result
1.  **Check for `sat`:** If the first line of output is `sat`, Z3 has found a state where the protocol is insolvent.
2.  **Inspect the Model:** The subsequent output block `(model ...)` provides the exact numbers (witness) that break the system. 
    - e.g., If `honest_deposit` is 1000 and `operator_earnings` is 50, but `total_assets` is only 1010, the proof has demonstrated a 40 SSV deficit.

---

## 2. Lean 4 Universal Theorem
**File:** `formal-proofs/ssv_global_insolvency_proof.lean`

### What it is
A formal mathematical proof written in Lean 4, a high-assurance functional programming language and theorem prover used by mathematicians and computer scientists.

### What it proves
It proves the **Insolvency Equivalence Theorem**. It proves that protocol insolvency is not just "possible" but is **mathematically guaranteed** if a cluster remains insolvent for longer than the duration its initial deposit can cover. It provides a universal proof that applies to *all* possible values, not just specific examples.

### How to compile
You need a working Lean 4 environment (Mathlib is required):
```bash
# From the root of a Lean project (like the one provided in this PoC)
lake env lean formal-proofs/ssv_global_insolvency_proof.lean
```

### How to verify the result
1.  **No Error Output:** In Lean, a successful proof produces **no output** (silent success). 
2.  **Kernel Verification:** If the file compiles, the Lean kernel has verified that every step of the reasoning follows the fundamental laws of logic.
3.  **Lemma Check:** The file includes a lemma `ssv_insolvency_foundry_witness`. By proving this, Lean confirms that the specific values used in our Foundry trace are mathematically certain to cause insolvency.

---

## 3. Z3-Python Global Trace
**File:** `scripts/verify_ssv_global_insolvency.py`

### What it is
A Python script that utilizes the `z3-solver` library to perform a high-level symbolic trace of the protocol's shared pool.

### What it proves
It specifically demonstrates the **Cross-User Fund Theft**. It shows how the uncollateralized virtual debt from one bankrupt user is fulfilled using the collateral of a second, honest user.

### How to execute
```bash
pip install z3-solver
python scripts/verify_ssv_global_insolvency.py
```

### How to verify the result
1.  **Console Narrative:** The script prints a human-readable "Trace Analysis."
2.  **Deficit Calculation:** Look for the line `=> Protocol Deficit: [X] SSV`. 
3.  **Code Mapping:** The script explicitly prints the mismatched lines of code in `OperatorLib.sol` and `ClusterLib.sol` that correspond to the logic being solved.

---

## Summary of Truth
| Tool | Nature of Proof | Outcome |
|------|-----------------|---------|
| **Z3 (SMT)** | Automated Search | Found a state where liabilities > assets. |
| **Lean 4** | Universal Math | Proved that the accounting mismatch *always* leads to insolvency over time. |
| **Foundry** | Execution Trace | Proved the bug exists in the actual mainnet bytecode. |

These three independent methodologies converge on a single, undeniable conclusion: **The ssv.network treasury is at risk of cascading insolvency.**
