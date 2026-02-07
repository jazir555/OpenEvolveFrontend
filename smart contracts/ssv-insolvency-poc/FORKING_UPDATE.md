# Forking Update - Mainnet Testing Compliance

**Date:** February 6, 2026  
**Purpose:** Update PoC to use proper mainnet forking per Immunefi guidelines

---

## Summary

The PoC has been updated to use `vm.createSelectFork()` instead of `vm.mockCall()` for testing. This ensures compliance with Immunefi's specific guideline:

> "The smart contract PoC should always be made by forking the mainnet using tools like Hardhat or Foundry. If forking the mainnet state is not feasible, using the project's existing test suite is an acceptable alternative. However, the test conditions must accurately reflect the state of the deployed code."

---

## Why Forking Matters

### Before (Mock Calls)
```solidity
function setUp() public {
    // Setup mock token behavior
    vm.mockCall(SSV_TOKEN, abi.encodeWithSignature("decimals()"), abi.encode(uint8(18)));
    vm.mockCall(SSV_TOKEN, abi.encodeWithSignature("balanceOf(address)"), abi.encode(uint256(0)));
    
    // Deploy the attack contract
    attackContract = new SSVInsolvencyPoC();
}
```

**Problem:** Mock calls simulate behavior but don't prove the vulnerability exists in actual deployed contracts. For a Critical bounty, the PoC must demonstrate the bug against real mainnet contracts.

### After (Mainnet Forking)
```solidity
function setUp() public {
    // Fork mainnet at a recent block
    vm.createSelectFork("mainnet", 19000000);
    
    // Deploy the attack contract
    attackContract = new SSVInsolvencyPoC();
    
    // Give attacker SSV tokens using actual token contract
    deal(SSV_TOKEN, address(attackContract), 1010e18);
    
    console.log(">>> Testing against actual SSV Network contracts on forked mainnet");
}
```

**Benefit:** Tests against the actual deployed SSV Network contracts, proving the vulnerability exists in production code.

---

## Changes Made

### 1. Test File (`test/SSVInsolvencyPoC.t.sol`)
- Replaced `vm.mockCall()` with `vm.createSelectFork("mainnet", 19000000)`
- Uses `deal()` cheatcode to provide test tokens on the forked state
- Tests against real contract addresses:
  - `SSVNetwork`: 0xDD9BC35aE942eF0cFa76930954a156B3fF30a4E1
  - `SSVNetworkViews`: 0xAE84579133f50A51E363cc00B5828f6C941C9Ce2
  - `SSVToken`: 0x9D65fF81a3c488d585bBfb0Bfe3c7707c7917f54

### 2. Foundry Configuration (`foundry.toml`)
- Added RPC endpoint configuration:
```toml
[rpc_endpoints]
mainnet = "${MAINNET_RPC_URL}"
```
- Added comments explaining Anvil alternative for local testing

### 3. Documentation Updates
- `README.md`: Updated safety notice to clarify forking is used
- `FINAL_VERIFICATION_REPORT.md`: Updated to reflect forking approach
- `SUBMISSION_GUIDE.md`: Added RPC setup instructions
- `POC_COMPLIANCE_REPORT.md`: Updated compliance verification
- `GUIDELINE_COMPLIANCE_CHECKLIST.md`: Clarified RPC requirement

---

## Running the Updated PoC

### Prerequisites
1. Foundry installed (`forge --version`)
2. RPC endpoint for Ethereum mainnet (Alchemy, Infura, etc.)

### Setup
```bash
# Set your RPC endpoint
export MAINNET_RPC_URL="https://eth-mainnet.g.alchemy.com/v2/YOUR_API_KEY"

# Or use Infura
export MAINNET_RPC_URL="https://mainnet.infura.io/v3/YOUR_PROJECT_ID"
```

### Run Tests
```bash
# Navigate to project
cd "smart contracts/ssv-insolvency-poc"

# Install dependencies
forge install

# Run tests
forge test -vv --match-path test/SSVInsolvencyPoC.t.sol
```

### Alternative: Using Anvil
If you prefer to run your own fork node:
```bash
# Terminal 1: Start Anvil with fork
anvil --fork-url https://eth-mainnet.g.alchemy.com/v2/YOUR_KEY

# Terminal 2: Run tests against local Anvil
export MAINNET_RPC_URL="http://localhost:8545"
forge test -vv --match-path test/SSVInsolvencyPoC.t.sol
```

---

## Safety Assurances

**Is this safe? Will I accidentally send transactions to mainnet?**

✅ **YES, this is completely safe.**

- `vm.createSelectFork()` creates a **local copy** of mainnet state
- All operations happen on your local machine only
- **NO transactions are sent to actual Ethereum mainnet**
- **NO transactions are sent to public testnets**
- The RPC is only used to **fetch the initial state** of contracts at block 19,000,000
- Once the fork is created, all testing is isolated and local

**Immunefi explicitly permits this approach** - it's the standard way to test against real contracts without risking actual funds.

---

## Compliance Verification

| Guideline | Status | Evidence |
|-----------|--------|----------|
| Fork mainnet | ✅ PASS | Uses `vm.createSelectFork("mainnet", 19000000)` |
| Test real contracts | ✅ PASS | Tests against actual SSV Network addresses |
| No public testing | ✅ PASS | Local fork only, no tx to mainnet |
| Accurate conditions | ✅ PASS | Tests deployed code at specific block |

---

## Files Modified

1. `test/SSVInsolvencyPoC.t.sol` - Updated to use forking
2. `src/SSVInsolvencyPoC.sol` - Updated safety comments
3. `foundry.toml` - Added RPC configuration
4. `README.md` - Updated documentation
5. `FINAL_VERIFICATION_REPORT.md` - Updated verification
6. `SUBMISSION_GUIDE.md` - Updated instructions
7. `POC_COMPLIANCE_REPORT.md` - Updated compliance info
8. `GUIDELINE_COMPLIANCE_CHECKLIST.md` - Updated checklist

---

## Questions?

**Q: Do I need a paid RPC endpoint?**  
A: Free tiers from Alchemy or Infura work fine for this PoC.

**Q: Will this use a lot of RPC credits?**  
A: No - the fork fetches state once at block 19,000,000. All subsequent operations are local.

**Q: Can I run this offline after the initial fork?**  
A: Yes - once the fork is created, you can disconnect from the internet and tests will continue to work.

**Q: Is this compliant with Immunefi rules?**  
A: Yes - this is the **recommended** approach by Immunefi guidelines.

---

*Updated: February 6, 2026*  
*Status: COMPLIANT WITH IMMUNEFI GUIDELINES*
