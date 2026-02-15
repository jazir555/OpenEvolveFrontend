# SSV Network Formal Verification - Implementation Guide

## Building the Project

### Prerequisites

1. Install Lean 4:
   ```bash
   curl https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh -sSf | sh
   ```

2. Install Lake (comes with Lean 4)

3. Clone the repository:
   ```bash
   cd lean_workspace/mathlib_project
   ```

### Building

```bash
# Build all modules
lake build

# Build specific module
lake build MathlibProject/Core/InsolvencyTheorem

# Clean and rebuild
lake clean && lake build
```

## Module Documentation

### Core.SSVTypes

Defines SSV-specific types:

```lean
def SSVAmount := Nat              -- SSV token amount in Wei
def BlockNumber := Nat            -- Ethereum block number
def ValidatorKey := Nat           -- Validator identifier
def OperatorId := Nat             -- Operator identifier
def ClusterSize := Nat            -- Number of operators in cluster
def BasisPoints := Nat            -- Fee percentage (10000 = 100%)
def LiquidationThreshold := Nat   -- Liquidation trigger (basis points)
```

### Core.ArithmeticSafety

Provides overflow-safe arithmetic:

```lean
def BoundedNat (max : Nat) := { n : Nat // n < max }
def boundedAdd {max : Nat} (a b : BoundedNat max) : BoundedNat max
def boundedMul {max : Nat} (a b : BoundedNat max) : BoundedNat max
```

**Key Theorem:** `ssv_supply_within_eth_bounds`
- Proves SSV supply (10^25 wei) < 2^256 - 1

### Core.OperatorFees

Calculates operator fees:

```lean
structure OperatorFeeConfig where
  feeBasisPoints : BasisPoints
  fee_valid : feeBasisPoints ≤ 10000
  min_fee : feeBasisPoints ≥ 0

def calculateOperatorFee (totalRewards : SSVAmount) (config : OperatorFeeConfig) : SSVAmount
def rewardsAfterFees (totalRewards : SSVAmount) (config : OperatorFeeConfig) : SSVAmount
```

**Key Theorem:** `operator_fee_bound`
- Proves operator fee ≤ total rewards

### Core.ClusterLiquidation

Manages cluster state and liquidation:

```lean
structure ClusterConfig where
  operatorCount : ClusterSize
  minOperators : operatorCount ≥ 4
  maxOperators : operatorCount ≤ 13
  liquidationThreshold : LiquidationThreshold
  threshold_valid : liquidationThreshold ≥ 8000

structure ClusterState where
  balance : SSVAmount
  virtualDebt : SSVAmount
  blocksElapsed : BlockNumber
  isLiquidated : Bool

def calculateHealthRatio (state : ClusterState) : Nat
def shouldLiquidate (config : ClusterConfig) (state : ClusterState) : Bool
def liquidateCluster (state : ClusterState) : ClusterState
```

**Key Theorems:**
- `health_ratio_bounded`: 0 ≤ health_ratio ≤ 10000
- `liquidation_zeros_debt`: Virtual debt becomes zero after liquidation

### Core.InsolvencyTheorem

Main insolvency proofs:

```lean
def calculateVirtualDebt (blocksElapsed : BlockNumber) (perBlockFee : SSVAmount) : SSVAmount
def calculateTotalLiabilities (balance : SSVAmount) (virtualDebt : SSVAmount) : SSVAmount
def isProtocolInsolvent (balance : SSVAmount) (virtualDebt : SSVAmount) : Bool
```

**Key Theorem:** `ssv_insolvency_possible`
- If balance > 0, blocks > 0, fee > 0, then liabilities > balance

### Utils.SSVHelpers

Helper functions:

```lean
def basisPointsToPercent (bp : BasisPoints) : Rat
def annualToPerBlockFee (annualFee : SSVAmount) : SSVAmount
def blocksUntilInsolvency (balance : SSVAmount) (perBlockFee : SSVAmount) : Nat
```

### Tests.PropertyTests

Property-based tests:

```lean
theorem property_fee_reduces_rewards
theorem property_virtual_debt_monotonic
theorem property_liabilities_monotonic
theorem property_liquidation_idempotent
```

## Proof Patterns

### Using linarith

For linear arithmetic:

```lean
example (a b : Nat) (h : a > 0) (h' : b > 0) : a * b > 0 := by
  linarith
```

### Using ring

For ring properties:

```lean
example (a b c : Nat) : a * (b + c) = a * b + a * c := by
  ring
```

### Using unfold and rw

For unfolding definitions:

```lean
theorem example_theorem (balance : Nat) :
  balance > 0 := by
  unfold balance
  linarith
```

### Using have

For introducing lemmas:

```lean
theorem complex_theorem (a b c : Nat) :
  a + b + c = a + (b + c) := by
  have h1 : a + b + c = (a + b) + c := by rfl
  have h2 : (a + b) + c = a + (b + c) := by ring
  rw [h1, h2]
```

## Extending the Proofs

### Adding a New Theorem

1. Define the types and functions in Core.SSVTypes
2. State the theorem in the appropriate Core file
3. Prove using Lean tactics
4. Add tests in Tests.PropertyTests

### Example: Adding a Slashing Mechanism

```lean
-- In Core.SSVTypes
def SlashAmount := Nat

-- In a new Core.Slashing.lean
def calculateSlash (misbehaviorCount : Nat) (balance : SSVAmount) : SlashAmount :=
  (balance * misbehaviorCount) / 100  -- 1% slash per misbehavior

theorem slash_bounded (misbehaviorCount : Nat) (balance : SSVAmount) :
  calculateSlash misbehaviorCount balance ≤ balance := by
  unfold calculateSlash
  have h_div : (balance * misbehaviorCount) / 100 ≤ balance * 100 / 100 := by
    apply Nat.div_le_div_right
    exact Nat.mul_le_mul_left balance (Nat.le_trans (Nat.zero_le 100) (by norm_num))
  have h_simpl : balance * 100 / 100 = balance := by
    exact (Nat.mul_div_right balance 100).symm
  rw [h_simpl] at h_div
  exact h_div
```

## Troubleshooting

### Build Errors

If you get "unknown constant" errors:
- Ensure Mathlib is synced: `lake update`
- Clean build: `lake clean && lake build`

### Proof Failures

If a proof fails:
1. Check definitions are correct
2. Use `lean --make file.lean` for better error messages
3. Simplify the proof and build up gradually

### Performance

For faster builds:
- Use `lake build -jN` for N parallel jobs
- Cache olean files (created automatically)

## Best Practices

1. **Document Everything:** Add docstrings to all theorems
2. **Use Namespaces:** Organize code by functionality
3. **Test Thoroughly:** Add property tests for all key theorems
4. **Keep Proofs Simple:** Prefer clarity over cleverness
5. **Verify Often:** Run `lake build` frequently to catch errors early

## Resources

- [Lean 4 Documentation](https://leanprover.github.io/lean4/doc/)
- [Mathlib4 Documentation](https://leanprover-community.github.io/mathlib4_docs/)
- [Lean Prover Community](https://leanprover-community.github.io/)
- [SSV Network Documentation](https://docs.ssv.network/)
