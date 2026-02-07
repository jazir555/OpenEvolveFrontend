# Immunefi Submission Guide - SSV Network Insolvency PoC

## Submission Methods

Per Immunefi's PoC Guidelines, you have **two options** for submitting the PoC:

### Option 1: Google Drive Upload (RECOMMENDED)

Since this PoC contains multiple files and configuration, **Google Drive is the recommended method**.

**Steps:**
1. Create a ZIP file of the `ssv-insolvency-poc/` directory
2. Upload to Google Drive
3. Set sharing to "Anyone with the link can view"
4. Submit the link in the Immunefi Dashboard

```bash
# Create ZIP file for upload
zip -r ssv-insolvency-poc.zip ssv-insolvency-poc/
```

**Upload Contents:**
```
ssv-insolvency-poc.zip
├── README.md
├── hardhat.config.js
├── package.json
├── contracts/InsolvencyPoC.sol
├── test/exploit.test.ts
├── scripts/*.py
└── formal-proofs/*
```

---

### Option 2: Paste Code in Submission (For Simple PoCs Only)

**Only use this if** the PoC is simple enough to fit in the submission comment.

**Not recommended for this vulnerability** because:
- Multiple files required
- Hardhat configuration needed
- Formal proofs are separate files

---

## Step-by-Step Submission Process

### Step 1: Prepare Your PoC Package

```bash
# Navigate to the PoC directory
cd "smart contracts/ssv-insolvency-poc"

# Create a clean ZIP (exclude node_modules if present)
zip -r ../ssv-insolvency-poc-submission.zip . -x "node_modules/*" "*.zip"
```

### Step 2: Upload to Google Drive

1. Go to https://drive.google.com
2. Upload `ssv-insolvency-poc-submission.zip`
3. Right-click → **Share**
4. Change to **"Anyone with the link can view"**
5. Copy the shareable link

### Step 3: Submit via Immunefi Dashboard

1. Go to https://immunefi.com/bug-bounty/ssvnetwork/
2. Click **"Submit a Bug"**
3. Log in to your Immunefi account
4. Fill out the submission form:

#### Submission Form Fields:

**Title:**
```
Critical: Protocol Insolvency via Uncollateralized Virtual Accounting Enables Direct Theft of User Funds
```

**Severity:** 
- Select: **Critical**
- Impact: **Protocol Insolvency**

**Description:**
```markdown
## Summary
The ssv.network protocol contains a critical accounting flaw where operator and DAO earnings accumulate unconditionally while cluster balances are capped at zero. This creates a state of protocol insolvency where virtual liabilities exceed actual assets, enabling direct theft of honest user deposits.

## Vulnerability Details
[Include summary from README.md Section 2]

## Impact
- Protocol Insolvency (Critical)
- Direct Theft of User Funds (Critical)
- Systemic Risk: Bank run scenario

## Proof of Concept
See attached Google Drive link for complete PoC including:
- Hardhat test demonstrating the exploit
- Formal mathematical proofs (Z3, Lean 4)
- Execution trace simulation
- Isolated vulnerable contract logic

## Funds at Risk
- TVL: ~60,600 SSV (~$215,130 USD)
- All user deposits in shared pool are at risk

## Remediation
Operator and DAO fee accumulation must be linked to cluster solvency.
```

**PoC Link:**
```
https://drive.google.com/file/d/[YOUR_FILE_ID]/view?usp=sharing
```

### Step 4: Additional Submission Details

**Attack Scenario:** (Paste from execution trace)
```
1. User A deposits 1000 SSV, User B deposits 10 SSV
2. User B's cluster goes bankrupt after 10 blocks
3. Operator withdraws 50 SSV of uncollateralized virtual earnings
4. User A can only withdraw 960 SSV (LOSS: 40 SSV)
```

**Suggested Fix:**
```markdown
Link operator/DAO earnings to cluster solvency by:
1. Only crediting operator fees when clusters have sufficient balance
2. Implementing global collateral check before withdrawals
3. Or segregating funds to prevent cross-cluster liability
```

---

## What Happens After Submission

### Timeline

| Stage | Timeline | What Happens |
|-------|----------|--------------|
| **Triage** | 24-48 hours | Immunefi reviews submission validity |
| **Project Review** | 7-30 days | SSV Network team validates the bug |
| **Negotiation** | Variable | Bounty amount discussion |
| **Payment** | First half of next month | SSV pays after DAO approval |

### Requirements for Payout

Per the bounty instructions:

1. **KYC Required:** You must provide:
   - Government ID (Passport or ID Card)
   - Proof of address (utility bill, bank statement)

2. **Bug Report Approval:**
   - SSV DAO Grants committee must approve disclosure
   - Payment sent first half of month following approval

3. **Bounty Calculation:**
   - 10% of funds at risk: $21,513
   - **Minimum applies: $50,000 USD**
   - Paid in SSV tokens at average price (CoinGecko + CoinMarketCap)

---

## Submission Checklist

Before submitting, verify:

- [ ] PoC ZIP file created and tested
- [ ] All files included (README, configs, tests, scripts)
- [ ] Google Drive link set to "Anyone with the link can view"
- [ ] Immunefi Dashboard form completed
- [ ] Severity selected: **Critical**
- [ ] Impact category: **Protocol Insolvency**
- [ ] TVL amount current in submission
- [ ] Vault address included: `0x2Be7549f1B58Fc3E81427a09E61e6D0B050A4C1D`

---

## Important Notes

### Do NOT:
- Test on mainnet or public testnet
- Disclose the bug publicly before approval
- Share the vulnerability details outside Immunefi
- Submit multiple reports for the same bug

### Do:
- Be responsive to questions from SSV team
- Provide additional information if requested
- Keep your PoC files backed up
- Be patient during the review process

### Communication

All communication happens through the **Immunefi Dashboard**. You'll receive email notifications when:
- Your report is triaged
- The project responds
- Bounty is approved
- Payment is processed

---

## Contact Information

- **Immunefi Support:** https://immunefi.com/support
- **SSV Network:** Through Immunefi Dashboard only
- **Bug Bounty Page:** https://immunefi.com/bug-bounty/ssvnetwork/

---

## Quick Reference

| Item | Details |
|------|---------|
| **Bounty Program** | SSV Network on Immunefi |
| **Max Bounty** | $1,000,000 USD |
| **Expected Bounty** | $50,000 USD (minimum) |
| **Vulnerability** | Protocol Insolvency |
| **Severity** | Critical |
| **PoC Method** | Google Drive Upload |
| **KYC Required** | Yes |
| **Payment Token** | SSV |
| **Vault Address** | 0x2Be7549f1B58Fc3E81427a09E61e6D0B050A4C1D |

---

## Example Google Drive Sharing Settings

```
File: ssv-insolvency-poc-submission.zip
Sharing: Anyone with the link
Permission: Viewer (not editor)
Link: https://drive.google.com/file/d/1xxxxxXXxxXXXXxxxXXxXXxxXXx/view?usp=sharing
```

**Copy this link and paste it in the Immunefi Dashboard submission form.**

---

*Last Updated: February 2026*  
*PoC Version: 1.0.0*  
*Status: Ready for Submission*
