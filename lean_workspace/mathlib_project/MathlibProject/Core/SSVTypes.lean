import Mathlib.Data.Nat.Basic
import Mathlib.Data.Int.Basic
import Mathlib.Algebra.Group.Basic
import Mathlib.Tactic.Linarith

namespace MathlibProject.SSV

/-- SSV Token amount in Wei (256-bit unsigned integer) -/
def SSVAmount := Nat

/-- Block number for Ethereum -/
def BlockNumber := Nat

/-- Validator public key identifier -/
def ValidatorKey := Nat

/-- Operator identifier -/
def OperatorId := Nat

/-- Number of operators in a cluster -/
def ClusterSize := Nat

/-- Fee percentage (basis points: 10000 = 100%, 1000 = 10%) -/
def BasisPoints := Nat

/-- Liquidation threshold (basis points) -/
def LiquidationThreshold := Nat

end MathlibProject.SSV
