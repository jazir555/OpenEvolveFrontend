# Immunefi Submission Guide - SSV Network Insolvency PoC

## PoC Format: Foundry (Immunefi Template)

This PoC follows the **Immunefi Forge PoC Templates** format using Foundry.

**Template Reference:** https://github.com/immunefi-team/forge-poc-templates

---

## Submission Methods

### Option 1: GitHub Repository (RECOMMENDED)

Upload this PoC to a private GitHub repository and share the link in the Immunefi Dashboard.

**Advantages:**
- Version control
- Easy for project to review
- Can include all files

**Steps:**
1. Create a private GitHub repo
2. Push all files
3. Share the repo link in submission

---

### Option 2: Google Drive Upload

If you prefer not to use GitHub, upload a ZIP file to Google Drive.

**Steps:**
```bash
# Create ZIP (exclude unnecessary files)
zip -r ssv-insolvency-poc-submission.zip . -x "*.zip" ".git/*"
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
forge test -vv --match-path test/SSVInsolvencyPoC.t.sol
```

**Expected output:** All tests pass with vulnerability demonstrated.

### Step 2: Prepare Submission Package

Ensure these files are included:

```
ssv-insolvency-poc/
├── foundry.toml              # Foundry config
├── README.md                 # Documentation
├── src/
│   ├── PoC.sol              # Immunefi base contract
│   ├── SSVInsolvencyPoC.sol # Attack contract
│   ├── log/                 # Logging utilities
│   └── tokens/              # Token utilities
├── test/
│   └── SSVInsolvencyPoC.t.sol # Test file
├── scripts/                  # Python verification scripts
└── formal-proofs/            # Formal verification files
```

### Step 3: Submit via Immunefi Dashboard

**URL:** https://immunefi.com/bug-bounty/ssvnetwork/

**Form Fields:**

| Field | Value |
|-------|-------|
| **Title** | Critical: Protocol Insolvency via Uncollateralized Virtual Accounting Enables Direct Theft of User Funds |
| **Severity** | Critical |
| **Impact** | Protocol Insolvency |
| **PoC Link** | [GitHub repo or Google Drive link] |

**Description Template:**

```markdown
## Summary
The ssv.network protocol contains a critical accounting flaw where operator and DAO 
earnings accumulate unconditionally while cluster balances are capped at zero. This 
creates protocol insolvency where virtual liabilities exceed actual assets.

## Vulnerability Details
- **Type:** Accounting Mismatch / Protocol Insolvency
- **Affected Files:** OperatorLib.sol, ProtocolLib.sol, ClusterLib.sol
- **Root Cause:** Unconditional operator/DAO credit vs capped cluster debit

## Impact
- Protocol Insolvency (Critical)
- Direct Theft of User Funds (Critical)
- All user deposits at risk

## Proof of Concept
Foundry-based PoC demonstrating the vulnerability:

```bash
git clone [YOUR_REPO]
cd ssv-insolvency-poc
forge install
forge test -vv --match-path test/SSVInsolvencyPoC.t.sol
```

Expected result: Test shows User A losing 40 SSV to operator theft.

## Formal Verification
- Z3 SMT-LIB proof (sat)
- Lean 4 mathematical theorems
- Python execution traces

## Funds at Risk
- TVL: ~60,600 SSV (~$215,130 USD)
- All deposits in shared pool at risk

## Suggested Fix
Link operator/DAO earnings to cluster solvency.
```

---

## Running the PoC (For Reviewers)

### Prerequisites
- [Foundry](https://book.getfoundry.sh/getting-started/installation)

### Commands

```bash
# Clone and setup
git clone <repo-url>
cd ssv-insolvency-poc
forge install

# Set RPC for mainnet forking (required per Immunefi guidelines)
export MAINNET_RPC_URL="https://eth-mainnet.g.alchemy.com/v2/YOUR_API_KEY"

# Run all tests
forge test -vv

# Run specific test
forge test -vv --match-test testInsolvencyAttack

# Run with full trace
forge test -vvv --match-test testInsolvencyAttack
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

This PoC follows the Immunefi Foundry template structure:

### Base Contract (`src/PoC.sol`)
- Snapshot functionality for balance tracking
- Logging utilities
- Profit calculation

### Attack Contract (`src/SSVInsolvencyPoC.sol`)
- Extends `PoC`
- Implements `initiateAttack()` function
- Demonstrates the vulnerability step-by-step

### Test Contract (`test/SSVInsolvencyPoC.t.sol`)
- Extends `PoC`
- Uses `setUp()` for initialization
- Uses `snapshot` modifier for balance tracking
- Tests prefixed with `test`

---

## Verification Checklist

Before submitting, verify:

- [ ] `forge build` succeeds
- [ ] `forge test -vv` passes (requires MAINNET_RPC_URL for forking)
- [ ] All tests demonstrate vulnerability
- [ ] README is clear and complete
- [ ] TVL amount is current
- [ ] GitHub repo is private (or Google Drive link ready)

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
| **Impact** | Protocol Insolvency |
| **Expected Bounty** | $50,000 USD (minimum) |
| **TVL at Risk** | ~$215,130 USD |
| **KYC Required** | Yes |
| **Payment Token** | SSV |

---

## Files Overview

| File | Purpose | Format |
|------|---------|--------|
| `src/SSVInsolvencyPoC.sol` | Attack contract | Foundry/Solidity |
| `test/SSVInsolvencyPoC.t.sol` | Test file | Foundry/Solidity |
| `formal-proofs/*.smt2` | Z3 proofs | SMT-LIB |
| `formal-proofs/*.lean` | Lean proofs | Lean 4 |
| `scripts/*.py` | Python verification | Python 3 |
| `README.md` | Documentation | Markdown |

---

*Last Updated: February 2026*  
*PoC Version: 1.0.0*  
*Foundry Version: ^0.8.13*
