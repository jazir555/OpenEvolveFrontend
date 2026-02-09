# SSV Network Liquidation Griefing PoC

## Overview

This is the **third attack vector** demonstrating the SSV Network protocol insolvency vulnerability. This PoC specifically targets the liquidation mechanism and demonstrates how an attacker can maximize theft by griefing liquidators.

> ⚠️ **SAFETY NOTICE:** This PoC operates entirely on a **local fork** of mainnet using Foundry's `vm.createSelectFork()`. No transactions are sent to actual mainnet.

**Attack Vector:** Time-Delayed Liquidation Griefing  
**Vulnerability:** Uncollateralized Virtual Accounting  
**Severity:** Critical  
**Impact:** Direct theft of user funds via maximized virtual debt  

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
cd ssv-insolvency-poc

# Run liquidation griefing test
forge test -vv --match-path test/SSVLiquidationGriefing.t.sol

# With full trace
forge test -vvv --match-test testLiquidationGriefingAttack
```

---

## Expected Output

```
=================================================================
SSV NETWORK: TIME-DELAYED LIQUIDATION GRIEFING ATTACK
=================================================================
Vulnerability: Uncollateralized Virtual Accounting + Liquidation Delay
Attack Vector: Maximize virtual debt by griefing liquidators
Impact: Direct theft of user funds
Severity: CRITICAL
=================================================================

--- PHASE 1: Setup Multiple Clusters ---

Victim Large deposited:   10000 SSV
Victim Small 1 deposited: 100 SSV (bankrupts in 100 blocks)
Victim Small 2 deposited: 50 SSV (bankrupts in 50 blocks)
Victim Small 3 deposited: 25 SSV (bankrupts in 25 blocks)
Total pool balance:       10175 SSV

--- PHASE 4: LIQUIDATION GRIEFING ---

Attacker monitors mempool for liquidate() transactions...
Attacker front-runs with high gas or exhausts liquidators
Liquidation DELAYED by 200 blocks!

After 200 blocks of griefing:
  - Victim Small 1: BANKRUPT
  - Victim Small 2: BANKRUPT
  - Victim Small 3: BANKRUPT
  - Virtual debt accumulated: 485 SSV
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

The liquidation griefing allowed 485 SSV of
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

## Files

| File | Description |
|------|-------------|
| `src/SSVLiquidationGriefingPoC.sol` | Main exploit contract |
| `test/SSVLiquidationGriefing.t.sol` | Foundry test file |
| `SSV_LIQUIDATION_GRIEFING_POC.md` | This documentation |

---

## Comparison to Other Attack Vectors

| Vector | Method | Scale | Detection |
|--------|--------|-------|-----------|
| **Single-Cluster** | One bankrupt cluster | Small | Easy |
| **Multi-Cluster** | Multiple operators + DAO | Medium | Moderate |
| **Liquidation Griefing** | Delay + maximize debt | **Large** | Hard |

The liquidation griefing attack is the **most severe** because:
1. It maximizes the virtual debt accumulation
2. It's harder to detect (looks like "slow" liquidations)
3. It compounds with multiple clusters
4. It can be executed by anyone (not just operators)

---

## Mitigation

### Immediate
- Implement liquidation incentives that make griefing economically unviable
- Add time-weighted withdrawal penalties during insolvency periods

### Long-term
- Link operator/DAO earnings to real-time cluster solvency
- Implement segregated pools to prevent cross-cluster liability
- Add global collateral checks before withdrawals

---

## References

- **Base PoC:** `src/SSVInsolvencyPoC.sol`
- **Mainnet Bytecode PoC:** `test/SSVMainnetExploit.t.sol`
- **Vulnerability Report:** `SSV_INSOLVENCY_VULNERABILITY.md`

---

*PoC Version: 1.0.0*  
*Last Updated: February 2026*
