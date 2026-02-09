# PoC Format Updates - SSV Liquidation Griefing

**Document:** Tracks updates and improvements to the PoC format.  
**Version:** 1.0.0  
**Last Updated:** February 7, 2026

---

## Version History

### v1.0.0 - Initial Release (February 2026)

**Changes:**
- Initial PoC creation following Immunefi Forge Template
- Liquidation griefing attack demonstration
- Time-delayed exploitation (200+ blocks)
- Formal verification (Z3 + Lean 4)
- Complete documentation suite

**Features:**
- 1 wei griefing deposit
- 200+ block exploitation window
- 485 SSV maximized virtual debt
- Front-running liquidators

---

## Format Compliance

### Immunefi Forge Template

| Requirement | Status | Notes |
|-------------|--------|-------|
| `foundry.toml` | ✅ | Properly configured |
| `src/` directory | ✅ | Attack contracts |
| `test/` directory | ✅ | Test files |
| Test extends Test | ✅ | `contract SSVLiquidationGriefingPoC is Test` |
| `setUp()` function | ✅ | Initialization |
| Test functions prefixed | ✅ | `testLiquidationGriefing()` |

### Documentation Standards

| Document | Status | Purpose |
|----------|--------|---------|
| README.md | ✅ | Main documentation |
| FINAL_AUDIT_REPORT.md | ✅ | Compliance verification |
| SUBMISSION_GUIDE.md | ✅ | How to submit |
| SUBMISSION_CHECKLIST.md | ✅ | Pre-submission checks |
| GUIDELINE_COMPLIANCE_CHECKLIST.md | ✅ | Immunefi compliance |
| POC_COMPLIANCE_REPORT.md | ✅ | Detailed compliance |
| TVL_UPDATE_GUIDE.md | ✅ | TVL maintenance |
| POC_INDEX.md | ✅ | Multi-PoC navigation |

---

## Key Improvements Over Standard Template

### 1. Multi-Language Support

Standard Immunefi template is Solidity-only. This PoC adds:

| Language | Files | Purpose |
|----------|-------|---------|
| Python | `scripts/*.py` | Formal verification, analysis |
| JavaScript | `scripts/*.js` | Hardhat compatibility |
| Lean 4 | `formal-proofs/*.lean` | Mathematical theorems |
| SMT-LIB | `formal-proofs/*.smt2` | Z3 constraint solving |

### 2. Extended Documentation

| Addition | Benefit |
|----------|---------|
| Formal Proofs Guide | Helps reviewers understand math |
| TVL Update Guide | Maintains current TVL |
| PoC Index | Navigates all three PoCs |
| Griefing explanation | Shows time-delay mechanics |

### 3. Safety Features

| Feature | Implementation |
|---------|----------------|
| Safety warnings | All files include warnings |
| Local fork only | `vm.createSelectFork()` |
| No real transactions | Test tokens via `deal()` |
| Clear documentation | Multiple safety notices |

---

## Comparison to Other PoCs

| Aspect | PoC 1 | PoC 2 | PoC 3 (This) |
|--------|-------|-------|--------------|
| Base format | Immunefi Template | Immunefi Template | Immunefi Template |
| Time delay | No | No | **Yes (200+ blocks)** |
| Griefing | No | No | **Yes** |
| Virtual debt | ~10 SSV | ~550 SSV | **~485 SSV** |
| Formal proofs | Same | Same | Same |
| Documentation | Complete | Complete | Complete |

---

## Future Updates

### Planned Improvements

| Version | Feature | Status |
|---------|---------|--------|
| v1.1.0 | Interactive visualization | Planned |
| v1.2.0 | Different griefing strategies | Planned |
| v2.0.0 | Automated TVL updates | Planned |

### Maintenance

| Task | Frequency | Responsible |
|------|-----------|-------------|
| TVL Update | Weekly | Researcher |
| SSV Price Check | Daily | Researcher |
| Documentation Review | Monthly | Researcher |
| Test Run | Before submission | Researcher |

---

## Format Verification

To verify PoC format:

```bash
# Check directory structure
ls -la
ls -la src/
ls -la test/
ls -la scripts/
ls -la formal-proofs/

# Check compilation
forge build

# Check tests
export MAINNET_RPC_URL="..."
forge test -vv

# Check documentation
ls *.md
```

---

*Format Version: 1.0.0*  
*Compatible with: Immunefi Forge Template v1.0+*
