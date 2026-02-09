# Immunefi Submission Guide - SSV Multi-Cluster Insolvency PoC

## PoC Format: Foundry (Immunefi Template)

This PoC follows the **Immunefi Forge PoC Templates** format using Foundry.

**Template Reference:** https://github.com/immunefi-team/forge-poc-templates

---

## Overview

This PoC demonstrates the **Multi-Cluster Cascading Insolvency** attack vector on SSV Network, showing how multiple bankrupt clusters compound the protocol's insolvency and create bank run dynamics.

**Key Difference from PoC 1:** While PoC 1 shows a single cluster attack, this PoC demonstrates **systemic risk** - the vulnerability affects the ENTIRE protocol when multiple clusters go bankrupt.

---

## Submission Methods

### Option 1: GitHub Repository (RECOMMENDED)

Upload this PoC to a private GitHub repository and share the link in the Immunefi Dashboard.

**Steps:**
1. Create a private GitHub repo
2. Push all files (including `lib/` submodules)
3. Share the repo link in submission

### Option 2: Google Drive Upload

```bash
# Create ZIP (exclude unnecessary files)
zip -r ssv-poc2-multi-cluster-submission.zip . -x "*.zip" ".git/*" "cache/*" "out/*"
```

1. Upload to Google Drive
2. Set sharing to "Anyone with the link can view"
3. Submit the link

---

## Step-by-Step Submission Process

### Step 1: Verify PoC Works

```bash
# Install dependencies
forge install

# Build
forge build

# Run tests
forge test -vv --match-path test/SSVMultiClusterInsolvency.t.sol
```

**Expected output:** All tests pass demonstrating multi-cluster insolvency.

### Step 2: Prepare Submission Package

**Required Files:**
```
ssv-poc2-multi-cluster/
├── foundry.toml              # Foundry config
├── README.md                 # Documentation
├── src/
│   └── SSVMultiClusterInsolvency.sol  # Attack contract
├── test/
│   └── SSVMultiClusterInsolvency.t.sol # Test file
├── scripts/                  # Python + JS verification
└── formal-proofs/            # Mathematical proofs
```

### Step 3: Submit via Immunefi Dashboard

**URL:** https://immunefi.com/bug-bounty/ssvnetwork/

**Form Fields:**

| Field | Value |
|-------|-------|
| **Title** | Critical: Multi-Cluster Cascading Insolvency Creates Bank Run Dynamics in SSV Network |
| **Severity** | Critical |
| **Impact** | Protocol Insolvency / Bank Run |
| **PoC Link** | [GitHub repo or Google Drive link] |

**Description Template:**

```markdown
## Summary
This PoC demonstrates that the SSV Network vulnerability is SYSTEMIC. When multiple 
clusters go bankrupt, the virtual debt compounds, creating a bank run scenario where 
early withdrawers (operators/DAO) profit at the expense of honest users.

## Vulnerability Details
- **Type:** Multi-Cluster Protocol Insolvency / Bank Run
- **Affected Files:** OperatorLib.sol, ProtocolLib.sol, ClusterLib.sol
- **Root Cause:** Unconditional operator/DAO credit vs capped cluster debit
- **Scale:** Affects entire protocol, not just individual clusters

## Impact
- Protocol Insolvency (Critical)
- Bank Run Dynamics (Critical)
- Systemic Risk to All Depositors (Critical)
- Direct Theft via Virtual Debt Accumulation

## Proof of Concept
Foundry-based PoC demonstrating multi-cluster attack:

```bash
git clone [YOUR_REPO]
cd ssv-poc2-multi-cluster
forge install
forge test -vv --match-path test/SSVMultiClusterInsolvency.t.sol
```

Expected result: 
- 3 clusters go bankrupt
- 550 SSV virtual debt created
- Large User loses 550 SSV to operators/DAO

## Formal Verification
- Z3 SMT-LIB proof (sat)
- Lean 4 mathematical theorems
- Python + JavaScript execution traces

## Funds at Risk
- TVL: ~60,600 SSV (~$215,130 USD)
- All deposits in shared pool at risk
- Risk COMPOUNDS with more bankrupt clusters

## Suggested Fix
Link operator/DAO earnings to cluster solvency.
Implement segregated pools to prevent cross-cluster liability.
```

---

## Running the PoC (For Reviewers)

### Prerequisites
- [Foundry](https://book.getfoundry.sh/getting-started/installation)
- Git

### Commands

```bash
# Clone and setup
git clone <repo-url>
cd ssv-poc2-multi-cluster
forge install

# Set RPC for mainnet forking (required per Immunefi guidelines)
export MAINNET_RPC_URL="https://eth-mainnet.g.alchemy.com/v2/YOUR_KEY"

# Run all tests
forge test -vv

# Run specific test
forge test -vv --match-test testMultiClusterInsolvency

# Run with full trace
forge test -vvv --match-test testMultiClusterInsolvency
```

**Note:** The PoC forks mainnet to test against actual SSV Network contracts as required by Immunefi guidelines.

---

## What Happens After Submission

### Timeline

| Stage | Timeline | Description |
|-------|----------|-------------|
| **Triage** | 24-48 hours | Immunefi validates submission |
| **Project Review** | 7-30 days | SSV Network team reviews |
| **Negotiation** | Variable | Bounty discussion |
| **Payment** | Next month | After DAO approval |

### Payout Requirements

1. **KYC Verification:**
   - Government ID (Passport/ID Card)
   - Proof of address (utility bill/bank statement)

2. **DAO Approval:**
   - SSV DAO Grants committee approval required
   - Payment first half of following month

### Expected Bounty

| Metric | Value |
|--------|-------|
| Funds at Risk | $215,130 USD |
| 10% of Funds | $21,513 |
| **Minimum Bounty** | **$50,000 USD** |
| Maximum Bounty | $1,000,000 USD |
| Payment Token | SSV |
| Price Source | CoinGecko + CoinMarketCap average |

---

## PoC Structure (Immunefi Template)

### Attack Contract (`src/SSVMultiClusterInsolvency.sol`)
- Extends `Test` (Forge-std)
- Demonstrates multi-cluster attack
- Shows bank run dynamics
- Calculates compounded virtual debt

### Test Contract (`test/SSVMultiClusterInsolvency.t.sol`)
- Extends `Test`
- Uses `setUp()` for initialization
- Tests prefixed with `test`
- Includes assertions for verification

---

## Verification Checklist

Before submitting, verify:

- [ ] `forge build` succeeds
- [ ] `forge test -vv` passes (requires MAINNET_RPC_URL for forking)
- [ ] All tests demonstrate vulnerability
- [ ] README is clear and complete
- [ ] TVL amount is current
- [ ] GitHub repo is private (or Google Drive link ready)
- [ ] Multi-cluster dynamics explained
- [ ] Bank run scenario documented

---

## Comparison to PoC 1

| Aspect | PoC 1 (Single-Cluster) | PoC 2 (Multi-Cluster) |
|--------|----------------------|----------------------|
| Scale | Small | **Large** |
| Clusters | 1 | 3 |
| Virtual Debt | ~10 SSV | ~550 SSV |
| Dynamics | Simple theft | Bank run |
| Systemic Risk | No | **Yes** |

**This PoC should be submitted ALONGSIDE PoC 1** to show the full scope of the vulnerability.

---

## Contact & Support

- **Immunefi Dashboard:** https://immunefi.com/dashboard
- **SSV Bounty Page:** https://immunefi.com/bug-bounty/ssvnetwork/
- **Immunefi Support:** https://immunefi.com/support

---

## Quick Reference

| Item | Details |
|------|---------|
| **Framework** | Foundry |
| **Template** | Immunefi Forge PoC Templates |
| **Severity** | Critical |
| **Impact** | Protocol Insolvency / Bank Run |
| **Expected Bounty** | $50,000 USD (minimum) |
| **TVL at Risk** | ~$215,130 USD |
| **KYC Required** | Yes |
| **Payment Token** | SSV |

---

## Files Overview

| File | Purpose | Format |
|------|---------|--------|
| `src/SSVMultiClusterInsolvency.sol` | Attack contract | Foundry/Solidity |
| `test/SSVMultiClusterInsolvency.t.sol` | Test file | Foundry/Solidity |
| `formal-proofs/*.smt2` | Z3 proofs | SMT-LIB |
| `formal-proofs/*.lean` | Lean proofs | Lean 4 |
| `scripts/*.py` | Python verification | Python 3 |
| `scripts/*.js` | JavaScript verification | Node.js |

---

*Last Updated: February 2026*  
*PoC Version: 1.0.0*  
*Foundry Version: ^0.8.13*
