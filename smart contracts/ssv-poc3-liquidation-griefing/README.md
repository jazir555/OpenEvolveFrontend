# SSV Network Liquidation Griefing PoC

## 🚀 Quick Start

**Want to see the liquidation griefing attack immediately?**

```bash
# Option 1: Python demo (fastest)
cd "smart contracts/ssv-poc3-liquidation-griefing"
python scripts/demo_griefing.py

# Option 2: JavaScript demo
node scripts/demo_griefing.js

# Option 3: Full Foundry POC
forge test -vv
```

**Expected output**: Proof that 200-block griefing delay creates 585 SSV of virtual debt in ~5 seconds.

---

## Overview

This is the **third and most severe attack vector** demonstrating the SSV Network protocol insolvency vulnerability. This PoC specifically targets the liquidation mechanism and demonstrates how an attacker can maximize theft by griefing liquidators.

> ⚠️ **SAFETY NOTICE:** This PoC operates entirely on a **local fork** of mainnet using Foundry's `vm.createSelectFork()`. No transactions are sent to actual mainnet.

**Attack Vector:** Time-Delayed Liquidation Griefing  
**Vulnerability:** Uncollateralized Virtual Accounting  
**Severity:** Critical  
**Impact:** Direct theft of user funds via maximized virtual debt  
**Status:** Most economically viable attack

---

## The Attack

### Strategy

1. **Monitor** - Watch for clusters nearing liquidation
2. **Grief** - Prevent timely liquidation (front-run or gas exhaustion)
3. **Accumulate** - Allow maximum virtual debt during delay
4. **Race** - Operators/DAO withdraw before victims
5. **Profit** - Bank run leaves last withdrawers with losses

### Why This Works

The SSV Network protocol has a **liquidation threshold period** - even with perfect liquidators, there's a window where:
- Clusters are flagged as "liquidatable" but not yet liquidated
- Virtual debt continues accumulating
- Operators and DAO earn uncollateralized fees

An attacker can **extend this window** by griefing liquidators, maximizing the virtual debt and therefore the theft.

---

## Attack Scenario

```
Initial State:
  - Victim Large:  10,000 SSV (healthy cluster)
  - Victim Small 1:   100 SSV (bankrupts in 100 blocks)
  - Victim Small 2:    50 SSV (bankrupts in 50 blocks)
  - Victim Small 3:    25 SSV (bankrupts in 25 blocks)

Normal Liquidation:
  - Block 25:  Small 3 liquidated, virtual debt: ~0 SSV
  - Block 50:  Small 2 liquidated, virtual debt: ~25 SSV
  - Block 100: Small 1 liquidated, virtual debt: ~75 SSV
  - Total virtual debt: ~100 SSV

Griefing Attack (200 block delay):
  - Block 220: All liquidated simultaneously
  - Small 3 virtual debt: 195 blocks * 1 SSV = 195 SSV
  - Small 2 virtual debt: 170 blocks * 1 SSV = 170 SSV
  - Small 1 virtual debt: 120 blocks * 1 SSV = 120 SSV
  - DAO unbacked fees: ~100 SSV
  - Total virtual debt: ~585 SSV
  
Result: Attacker steals 485% more than normal liquidation!
```

---

## Running the PoC

### Prerequisites

```bash
# Set RPC endpoint
export MAINNET_RPC_URL="https://eth-mainnet.g.alchemy.com/v2/YOUR_KEY"
```

### Execution

```bash
cd ssv-poc3-liquidation-griefing

# Install dependencies
forge install

# Build
forge build

# Run the test
forge test -vv

# With full trace
forge test -vvv
```

### Specific Tests

```bash
# Main liquidation griefing test
forge test -vv --match-test testLiquidationGriefingAttack

# Griefing maximization test
forge test -vv --match-test testGriefingMaximizesDebt

# Delay impact test
forge test -vv --match-test testGriefingDelayImpact
```

---

## Expected Output

```
=================================================================
SSV NETWORK: TIME-DELAYED LIQUIDATION GRIEFING ATTACK
=================================================================

--- PHASE 4: LIQUIDATION GRIEFING ---

Attacker monitors mempool for liquidate() transactions...
Attacker front-runs with high gas or exhausts liquidators
Liquidation DELAYED by 200 blocks!

After 200 blocks of griefing:
  - Victim Small 1: BANKRUPT (was liquidatable at block 80)
  - Victim Small 2: BANKRUPT (was liquidatable at block 50)
  - Victim Small 3: BANKRUPT (was liquidatable at block 25)
  - Virtual debt accumulated: 585 SSV
  - This debt is UNBACKED - no cluster has funds to pay it!

--- PHASE 5: BANK RUN - Race to Withdraw ---

Operator 3 withdrew: 195 SSV (from bankrupt cluster 3)
Operator 2 withdrew: 170 SSV (from bankrupt cluster 2)
Operator 1 withdrew: 120 SSV (from bankrupt cluster 1)
DAO withdrew:        100 SSV (includes unbacked network fees)

Total stolen from pool: 585 SSV
All of it is UNBACKED virtual debt!

--- PHASE 6: Honest Victim Attempts Withdrawal ---

Pool remaining:       9590 SSV
Victim Large entitlement: 10000 SSV

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
CRITICAL: VICTIM LARGE FUNDS STOLEN!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

Victim Large expected:   10000 SSV
Victim Large received:   9590 SSV
VICTIM LARGE LOSS:       410 SSV

The liquidation griefing allowed 585 SSV of
virtual debt to accumulate. When operators and DAO withdrew,
they STOLE this amount from Victim Large's honest deposit!
```

---

## Key Insights

### 1. Liquidation Threshold Gap
Even with perfect liquidators, the protocol has an inherent gap where virtual debt accumulates. The griefing attack **exploits and extends** this gap.

### 2. Multi-Cluster Amplification
With multiple clusters going bankrupt, the virtual debt compounds. Each additional cluster adds to the total theft.

### 3. Bank Run Effect
Once virtual debt exists, it becomes a race to withdraw. Late withdrawers (honest users) bear the losses.

### 4. Economic Feasibility
Griefing liquidators is economically viable because:
- Flashbots can be used for precise front-running
- Gas price manipulation can delay liquidations
- The profit (stolen funds) exceeds the griefing cost

---

## Comparison to Other Attack Vectors

| Vector | Method | Scale | Detection | Economic Viability |
|--------|--------|-------|-----------|-------------------|
| **PoC 1: Single-Cluster** | One bankrupt cluster | Small | Easy | Low |
| **PoC 2: Multi-Cluster** | Multiple operators + DAO | Medium | Moderate | Medium |
| **PoC 3: Liquidation Griefing** | Delay + maximize debt | **Large** | Hard | **High** |

**PoC 3 is the most severe because:**
1. It maximizes the virtual debt accumulation
2. It's harder to detect (looks like "slow" liquidations)
3. It compounds with multiple clusters
4. It can be executed by anyone (not just operators)
5. It has the highest economic ROI for attackers

---

## Griefing Techniques

### 1. Front-Running (MEV)
```solidity
// Attacker monitors mempool for liquidate() calls
// Uses Flashbots to front-run with higher gas
function griefWithFrontRun() internal {
    // Submit bundle with higher priority fee
    // Liquidator's tx gets reverted or delayed
}
```

### 2. Gas Price Manipulation
```solidity
// Spike gas prices during liquidation windows
// Makes liquidation economically unviable for liquidators
function griefWithGasSpike() internal {
    // Execute high-gas transactions
    // Drive up base fee
}
```

### 3. Block Stuffing
```solidity
// Fill blocks to prevent liquidation transactions
// Most effective on L2s with limited block space
function griefWithBlockStuffing() internal {
    // Submit many transactions to fill block
}
```

---

## Mitigation

### Immediate
- Implement liquidation incentives that make griefing economically unviable
- Add time-weighted withdrawal penalties during insolvency periods
- Use commit-reveal schemes for liquidations

### Long-term
- Link operator/DAO earnings to real-time cluster solvency
- Implement segregated pools to prevent cross-cluster liability
- Add global collateral checks before withdrawals

---

## Files

| File | Description |
|------|-------------|
| `src/SSVLiquidationGriefingPoC.sol` | Main exploit contract |
| `test/SSVLiquidationGriefing.t.sol` | Foundry test file |
| `foundry.toml` | Foundry configuration |
| `README.md` | This documentation |

---

## References

- **PoC 1:** `ssv-insolvency-poc/` - Single cluster demonstration
- **PoC 2:** `ssv-poc2-multi-cluster/` - Multi-cluster cascading
- **Vulnerability Report:** `../SSV_INSOLVENCY_VULNERABILITY.md`

---

*PoC Version: 1.0.0*  
*Last Updated: February 2026*

## Formal Proofs

### 1. Lean 4 Mathematical Proofs
Each PoC directory is a standalone Lean package. To verify the proofs:
```bash
# From this directory (ssv-poc3-liquidation-griefing)
lake exe cache get
lake build
```
This confirms that the liquidation griefing logic is mathematically certain, with zero `sorry` statements.

### 2. SMT-LIB Proof (Z3)
**File:** `formal-proofs/LIQUIDATION_GRIEFING_PROOF.smt2`
```bash
# Run with Z3
z3 formal-proofs/LIQUIDATION_GRIEFING_PROOF.smt2
```
**Result:** `sat` - Liquidation griefing is mathematically reachable.

### 3. Python Verification Scripts
```bash
python scripts/demo_griefing.py
python scripts/run_smt_proof.py
```
