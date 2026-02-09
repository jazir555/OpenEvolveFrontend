# Immunefi Submission Guide - SSV Liquidation Griefing PoC

## PoC Format: Foundry (Immunefi Template)

This PoC follows the **Immunefi Forge PoC Templates** format using Foundry.

**Template Reference:** https://github.com/immunefi-team/forge-poc-templates

---

## Overview

This PoC demonstrates the **Liquidation Griefing** attack vector on SSV Network, showing how an attacker can maximize virtual debt accumulation by griefing liquidators and extending the exploitation window.

**Key Difference from PoC 1 & 2:** While PoCs 1-2 show immediate exploitation, this PoC demonstrates **time-delayed exploitation** where virtual debt accumulates over **200+ blocks** through griefing attacks.

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
zip -r ssv-poc3-liquidation-griefing-submission.zip . -x "*.zip" ".git/*" "cache/*" "out/*"
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
forge test -vv --match-path test/SSVLiquidationGriefingPoC.t.sol
```

**Expected output:** All tests pass demonstrating liquidation griefing.

### Step 2: Prepare Submission Package

**Required Files:**
```
ssv-poc3-liquidation-griefing/
├── foundry.toml              # Foundry config
├── README.md                 # Documentation
├── src/
│   └── SSVLiquidationGriefingPoC.sol  # Attack contract
├── test/
│   └── SSVLiquidationGriefingPoC.t.sol # Test file
├── scripts/                  # Python + JS verification
└── formal-proofs/            # Mathematical proofs
```

### Step 3: Submit via Immunefi Dashboard

**URL:** https://immunefi.com/bug-bounty/ssvnetwork/

**Form Fields:**

| Field | Value |
|-------|-------|
| **Title** | Critical: Liquidation Griefing Attack Maximizes Protocol Insolvency in SSV Network |
| **Severity** | Critical |
| **Impact** | Protocol Insolvency / Time-Delayed Exploitation |
| **PoC Link** | [GitHub repo or Google Drive link] |

**Description Template:**

```markdown
## Summary
This PoC demonstrates that attackers can EXTEND the exploitation window and 
MAXIMIZE virtual debt through liquidation griefing. By front-running liquidators 
with minimal deposits, attackers keep insolvent clusters active for 200+ blocks, 
allowing operators to accumulate 485+ SSV of uncollateralized virtual debt.

## Vulnerability Details
- **Type:** Liquidation Griefing / Time-Delayed Exploitation
- **Affected Files:** LiquidationThreshold.sol, OperatorLib.sol, ClusterLib.sol
- **Root Cause:** Griefing liquidators extends insolvency window
- **Impact:** Virtual debt is maximized over extended time period

## Attack Steps
1. Find liquidatable cluster (below threshold but not yet liquidated)
2. Grief liquidators by front-running with 1 wei deposit
3. Wait 200+ blocks while cluster remains active but insolvent
4. Operators accumulate 485+ SSV of virtual debt
5. Operators withdraw first (race condition), leaving insufficient funds
6. Result: 485+ SSV deficit for honest users

## PoC
Run: forge test -vv --match-test testLiquidationGriefing

Expected result:
- Griefing extends exploitation window to 200+ blocks
- Virtual debt accumulates to 485 SSV
- Large User loses 485 SSV

## Formal Verification
- Z3 SMT-LIB proof (sat)
- Lean 4 mathematical theorems
- Python + JavaScript execution traces

## Funds at Risk
- TVL: ~60,600 SSV (~$215,130 USD)
- All deposits in shared pool at risk
- Griefing attack maximizes damage

## Suggested Fix
1. Implement griefing-resistant liquidation
2. Link operator/DAO earnings to cluster solvency
3. Implement segregated pools
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
cd ssv-poc3-liquidation-griefing
forge install

# Set RPC for mainnet forking (required per Immunefi guidelines)
export MAINNET_RPC_URL="https://eth-mainnet.g.alchemy.com/v2/YOUR_KEY"

# Run all tests
forge test -vv

# Run specific test
forge test -vv --match-test testLiquidationGriefing

# Run with full trace
forge test -vvv --match-test testLiquidationGriefing
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

### Attack Contract (`src/SSVLiquidationGriefingPoC.sol`)
- Extends `Test` (Forge-std)
- Demonstrates griefing attack
- Shows time-delayed exploitation
- Calculates accumulated virtual debt

### Test Contract (`test/SSVLiquidationGriefingPoC.t.sol`)
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
- [ ] Griefing attack clearly explained
- [ ] Time-delay dynamics documented

---

## Comparison to PoC 1 & 2

| Aspect | PoC 1 (Single) | PoC 2 (Multi) | PoC 3 (This - Griefing) |
|--------|---------------|---------------|------------------------|
| Timing | Immediate | Immediate | **Delayed (200+ blocks)** |
| Debt Accumulation | Fixed | Compounding | **Maximized via griefing** |
| Virtual Debt | ~10 SSV | ~550 SSV | **~485 SSV** |
| Attack Control | Limited | Limited | **Extended window** |

**This PoC should be submitted ALONGSIDE PoC 1 and 2** to show the full scope of the vulnerability.

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
| **Impact** | Protocol Insolvency / Time-Delayed Exploitation |
| **Expected Bounty** | $50,000 USD (minimum) |
| **TVL at Risk** | ~$215,130 USD |
| **KYC Required** | Yes |
| **Payment Token** | SSV |

---

## Files Overview

| File | Purpose | Format |
|------|---------|--------|
| `src/SSVLiquidationGriefingPoC.sol` | Attack contract | Foundry/Solidity |
| `test/SSVLiquidationGriefingPoC.t.sol` | Test file | Foundry/Solidity |
| `formal-proofs/*.smt2` | Z3 proofs | SMT-LIB |
| `formal-proofs/*.lean` | Lean proofs | Lean 4 |
| `scripts/*.py` | Python verification | Python 3 |
| `scripts/*.js` | JavaScript verification | Node.js |

---

*Last Updated: February 2026*  
*PoC Version: 1.0.0*  
*Foundry Version: ^0.8.13*
