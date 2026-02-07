# PoC: Systematic Protocol Insolvency in ssv.network

## 1. Title
**Systematic Protocol Insolvency via Uncollateralized Virtual Accounting leads to Direct Theft of User Funds**

## 2. Description

### Brief/Intro
The ssv.network protocol utilizes a "decoupled virtual credit" system for operator and DAO fees that is fundamentally insolvent by design. While individual cluster balances are correctly capped at zero during fee deductions, the corresponding rewards credited to operators and the DAO continue to accumulate unconditionally. This creates a state of **Protocol Insolvency** where the protocol's total liabilities (promised rewards + user deposits) exceed its actual SSV token holdings.

### Vulnerability Details
The vulnerability stems from a logical mismatch between reward accumulation and cluster debiting logic:

1.  **Unconditional Operator Credit:** In `OperatorLib.sol`, operator balances are increased unconditionally based on the passage of blocks and the operator's total `validatorCount`, without checking if the underlying clusters have a positive balance.
2.  **Unconditional DAO Credit:** In `ProtocolLib.sol`, the DAO’s earnings are calculated using a global index that ignores individual cluster solvency.
3.  **Capped Cluster Debit:** Conversely, `ClusterLib.sol` correctly caps the deduction from a cluster's balance at zero.

**The Logical Flaw:** When a cluster stays insolvent (bankrupt) but is not yet liquidated, the protocol continues to credit "Virtual SSV" to operators and the DAO. Because all SSV tokens are held in a shared contract pool, these virtual rewards are fulfilled using the tokens deposited by other, healthy clusters.

### Impact Details
*   **Protocol Insolvency (Critical):** The protocol overpromises assets it does not hold.
*   **Direct Theft of Funds (Critical):** Uncollateralized operator withdrawals are paid out using the principal deposits of honest users.
*   **Systemic Risk:** The protocol's solvency relies on 100% efficient liquidation. Any delay creates immediate, unrecoverable debt.

## 3. Proof of Concept

### Prerequisites
- Node.js and npm
- Hardhat

### Runnable Exploit Trace
The exploit demonstrates that an honest user (User A) is unable to withdraw their full deposit after an operator drains the pool by exploiting uncollateralized virtual debt from a bankrupt cluster (User B).

**Setup:**
1. Navigate to the `ssv-network` directory.
2. Copy `test/exploit.test.ts` (provided in this PoC) into the project's `test/` folder.
3. Run: `npx hardhat test test/exploit.test.ts`

**Key Result from Trace:**
- Initial Contract Balance: **5.1 SSV**
- 100M blocks pass; Cluster B becomes insolvent.
- 4 Operators withdraw their full virtual earnings (approx 0.2 SSV each).
- Total Withdrawn by Operators: **~0.8 SSV**.
- Final Contract Balance: **4.3 SSV**.
- **Deficit:** User A is entitled to **5.0 SSV** but the contract only has **4.3 SSV**.
- **Result:** User A has lost **0.7 SSV** of their principal.

### Formal Mathematical Evidence
We have included formal proofs in the `formal-proofs/` directory:
- `ssv_global_insolvency_proof.lean`: Universal Lean 4 proof that insolvency is inevitable.
- `SSV_INSOLVENCY_PROOF.smt2`: Z3 SMT-LIB model demonstrating reachability.
- `SSV_FORMAL_PROOF_CERTIFICATE.json`: Machine-readable proof certificate.

## 4. Amount of Funds at Risk

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

The vulnerability affects the *entire* shared pool of SSV tokens in the SSVNetwork 
contract (0xDD9BC35aE942eF0cFa76930954a156B3fF30a4E1), as any uncollateralized 
virtual debt is fulfilled from the total contract balance. All user deposits are 
at risk of partial or total loss due to protocol insolvency.
