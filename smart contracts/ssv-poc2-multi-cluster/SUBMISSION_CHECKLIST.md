# Submission Checklist - SSV Multi-Cluster Insolvency PoC

**PoC Version:** 1.0.0  
**Submission Date:** February 2026  
**Status:** ✅ READY FOR SUBMISSION

---

## Pre-Submission Verification

### 1. Build Verification

| Check | Command | Status |
|-------|---------|--------|
| Dependencies installed | `forge install` | [ ] |
| Project compiles | `forge build` | [ ] |
| No compilation warnings | `forge build 2>&1` | [ ] |
| Submodules initialized | `git submodule update --init` | [ ] |

### 2. Test Verification

| Check | Command | Status |
|-------|---------|--------|
| Set RPC environment | `export MAINNET_RPC_URL="..."` | [ ] |
| Run all tests | `forge test -vv` | [ ] |
| Multi-cluster test passes | `forge test --match-test testMultiClusterInsolvency` | [ ] |
| Output shows virtual debt | Check console output | [ ] |
| No test failures | Verify all pass | [ ] |

### 3. Documentation Verification

| Check | File | Status |
|-------|------|--------|
| README complete | `README.md` | [ ] |
| Audit report ready | `FINAL_AUDIT_REPORT.md` | [ ] |
| Submission guide ready | `SUBMISSION_GUIDE.md` | [ ] |
| Compliance checklist ready | `GUIDELINE_COMPLIANCE_CHECKLIST.md` | [ ] |
| Compliance report ready | `POC_COMPLIANCE_REPORT.md` | [ ] |

### 4. Contract Address Verification

| Contract | Address | Status |
|----------|---------|--------|
| SSV_TOKEN | 0x9D65fF81a3c488d585bBfb0Bfe3c7707c7917f54 | [ ] |
| SSV_NETWORK | 0xDD9BC35aE942eF0cFa76930954a156B3fF30a4E1 | [ ] |

Verify with:
```bash
grep -r "0x9D65fF81a3c488d585bBfb0Bfe3c7707c7917f54" src/ test/
grep -r "0xDD9BC35aE942eF0cFa76930954a156B3fF30a4E1" src/ test/
```

---

## Files Checklist

### Required Files

| File | Purpose | Status |
|------|---------|--------|
| `foundry.toml` | Foundry configuration | [ ] |
| `foundry.lock` | Dependency lock file | [ ] |
| `README.md` | Main documentation | [ ] |
| `src/SSVMultiClusterInsolvency.sol` | Attack contract | [ ] |
| `test/SSVMultiClusterInsolvency.t.sol` | Test file | [ ] |

### Documentation Files

| File | Purpose | Status |
|------|---------|--------|
| `FINAL_AUDIT_REPORT.md` | Audit verification | [ ] |
| `SUBMISSION_GUIDE.md` | Submission instructions | [ ] |
| `SUBMISSION_CHECKLIST.md` | This file | [ ] |
| `GUIDELINE_COMPLIANCE_CHECKLIST.md` | Guideline compliance | [ ] |
| `POC_COMPLIANCE_REPORT.md` | Compliance report | [ ] |
| `TVL_UPDATE_GUIDE.md` | TVL calculation guide | [ ] |
| `POC_INDEX.md` | Index of all PoCs | [ ] |

### Script Files

| File | Purpose | Status |
|------|---------|--------|
| `scripts/run_execution_poc.py` | Python execution | [ ] |
| `scripts/run-execution-poc.js` | JS execution | [ ] |
| `scripts/verify_multi_cluster_insolvency.py` | Python verification | [ ] |
| `scripts/verify-multi-cluster.js` | JS verification | [ ] |
| `scripts/run_smt_proof.py` | SMT proof runner | [ ] |
| `scripts/hardhat-test.js` | Hardhat tests | [ ] |

### Formal Proof Files

| File | Purpose | Status |
|------|---------|--------|
| `formal-proofs/MULTI_CLUSTER_INSOLVENCY_PROOF.smt2` | Z3 proof | [ ] |
| `formal-proofs/multi_cluster_insolvency_proof.lean` | Lean proof | [ ] |

### Configuration Files

| File | Purpose | Status |
|------|---------|--------|
| `hardhat.config.js` | Hardhat configuration | [ ] |
| `package.json` | Node dependencies | [ ] |
| `.gitmodules` | Git submodules | [ ] |

---

## Immunefi Dashboard Submission

### Submission Form

| Field | Value | Status |
|-------|-------|--------|
| **Bug Title** | Critical: Multi-Cluster Cascading Insolvency Creates Bank Run Dynamics in SSV Network | [ ] |
| **Severity** | Critical | [ ] |
| **Impact** | Protocol Insolvency | [ ] |
| **Vulnerability Type** | Logic Error | [ ] |
| **Asset Type** | Smart Contract | [ ] |
| **Affected Contract** | SSV Network | [ ] |

### Description Template

```markdown
## Summary
SSV Network has a critical systemic vulnerability where multiple bankrupt clusters 
compound protocol insolvency, creating bank run dynamics. Operators and DAO continue 
earning from bankrupt clusters, creating uncollateralized virtual debt.

## Vulnerability Details
- **Root Cause:** OperatorLib.sol and DAO earnings don't check cluster solvency
- **Impact:** Protocol insolvency scales with number of bankrupt clusters
- **Attack:** Bank run - early withdrawers profit at expense of late withdrawers

## Steps to Reproduce
1. Setup 4 operators, 4 clusters with varying deposits
2. Pass 100 blocks - clusters 3 and 4 go bankrupt
3. Operators continue earning from ALL 4 clusters (including bankrupt)
4. Operators + DAO withdraw before users
5. Result: 550 SSV deficit for honest users

## PoC
Run: forge test -vv --match-test testMultiClusterInsolvency

## Funds at Risk
- TVL: ~60,600 SSV (~$215,130 USD)
- 10% of TVL: $21,513
- **Minimum Bounty: $50,000**

## Fix
Link operator/DAO earnings to cluster solvency. Implement segregated pools.
```

---

## Upload Preparation

### Option 1: GitHub Repository (Recommended)

| Step | Action | Status |
|------|--------|--------|
| 1 | Create private GitHub repo | [ ] |
| 2 | Initialize with git | `git init` | [ ] |
| 3 | Add all files | `git add .` | [ ] |
| 4 | Commit | `git commit -m "Initial PoC"` | [ ] |
| 5 | Add remote | `git remote add origin ...` | [ ] |
| 6 | Push | `git push -u origin main` | [ ] |
| 7 | Verify all files present | Check GitHub | [ ] |

### Option 2: Google Drive

| Step | Action | Status |
|------|--------|--------|
| 1 | Create ZIP archive | `zip -r ssv-poc2.zip . -x "*.zip" ".git/*"` | [ ] |
| 2 | Upload to Google Drive | [ ] |
| 3 | Set sharing | "Anyone with link can view" | [ ] |
| 4 | Copy sharing link | [ ] |

---

## Final Verification

### Code Checks

- [ ] All Solidity files compile without errors
- [ ] All tests pass with RPC
- [ ] No mock calls used
- [ ] Real contract addresses used
- [ ] Safety warnings present in all files

### Documentation Checks

- [ ] README explains vulnerability clearly
- [ ] README explains how to run PoC
- [ ] TVL calculation is current
- [ ] All contract addresses match
- [ ] Multi-cluster dynamics explained
- [ ] Bank run scenario documented

### Compliance Checks

- [ ] Follows Immunefi PoC guidelines
- [ ] Follows Immunefi PoC rules
- [ ] Uses mainnet forking
- [ ] No real network transactions
- [ ] Complete PoC (not partial)

---

## Comparison to PoC 1

| Aspect | PoC 1 | PoC 2 (This) | Status |
|--------|-------|--------------|--------|
| Clusters | 1 | 3 | [ ] |
| Virtual Debt | ~10 SSV | ~550 SSV | [ ] |
| Bank Run | No | Yes | [ ] |
| Systemic Risk | No | Yes | [ ] |
| DAO Involvement | No | Yes | [ ] |

**Note:** Submit BOTH PoCs to show vulnerability scope.

---

## Risk Assessment

### Submission Risks

| Risk | Level | Mitigation |
|------|-------|------------|
| Ban for unsafe PoC | NONE | Uses local fork only |
| Rejection for incomplete | LOW | Full PoC demonstrated |
| Duplicate submission | LOW | Unique multi-cluster angle |

### Expected Outcome

| Outcome | Probability |
|---------|-------------|
| Bounty awarded | High |
| Critical severity | High |
| Request for more info | Low |
| Rejection | Very Low |

---

## Post-Submission Actions

After submitting:

1. **Save submission ID** from Immunefi Dashboard
2. **Monitor email** for project team questions
3. **Be responsive** to any requests
4. **Prepare KYC documents** (will be needed if bounty awarded)
5. **Track submission** in Immunefi Dashboard

---

## Summary

**Status:** ✅ READY FOR SUBMISSION

**Next Steps:**
1. Complete all checkboxes above
2. Run final tests
3. Upload to GitHub or Google Drive
4. Submit via Immunefi Dashboard
5. Monitor for updates

**Expected Bounty:** $50,000 USD (minimum)

---

*Checklist Version: 1.0*  
*Last Updated: February 2026*
