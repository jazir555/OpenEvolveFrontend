"""
Smart Contract Domain Knowledge

This module defines specific vulnerability patterns and "Attack Vectors" for smart contracts.
It uses the logic defined in `smart_contract_logic_analyzer.py` to construct
specific predicates that represent common hacks (Reentrancy, Flash Loan, etc.).

This acts as the "Database" for the Audit Engine.

Author: OpenEvolve
"""

import z3
from smart_contract_logic_analyzer import ContractState

class AttackVector:
    """Base class for an attack vector definition."""
    name: str = "Generic"
    severity: str = "Low"
    description: str = ""

    def define_predicate(self, state: ContractState) -> z3.ExprRef:
        """
        Define the Z3 predicate that must be satisfiable for this attack to succeed.
        Must be implemented by subclasses.
        """
        raise NotImplementedError

class ReentrancyAttack(AttackVector):
    name = "Reentrancy"
    severity = "Critical"
    description = "External call occurs before state update, allowing recursive calls to drain funds."

    def define_predicate(self, state: ContractState) -> z3.ExprRef:
        """
        Simplistic symbolic model of reentrancy:
        Exits a state where:
        1. Contract balance > 0
        2. 'locked' status is False (checking for non-reentrant modifier)
        3. An external call is made (symbolized by a boolean flag)
        4. Balance is NOT updated before call (Invariant violation)
        """
        # Declare specific variables for this pattern
        contract_balance = state.balances[state.contract_address]
        is_locked = state.declare_storage_var("reentrancy_lock", "bool")
        external_call_made = z3.Bool('external_call_executed')
        balance_deducted = z3.Bool('balance_deducted_before_call')
        
        # The vulnerability condition:
        # We can make a call, the lock is NOT active (or missing), 
        # and we haven't deducted the balance yet.
        # AND the contract has funds to steal.
        vulnerability = z3.And(
            contract_balance > 0,
            z3.Not(is_locked),
            external_call_made,
            z3.Not(balance_deducted)
        )
        return vulnerability

class IntegerOverflowAttack(AttackVector):
    name = "Integer Overflow"
    severity = "High"
    description = "Arithmetic operation exceeds variable storage capacity."

    def define_predicate(self, state: ContractState) -> z3.ExprRef:
        # Example: uint256 overflow
        # Input amount
        amount = z3.BitVec('input_amount', 256)
        current_balance = z3.BitVec('user_balance', 256)
        
        # Max uint256
        MAX_UINT = z3.BitVecVal(2**256 - 1, 256)
        
        # The condition: amount + current_balance < amount (classic overflow wraparound)
        # Note: Z3's BitVec handles overflow natively, so checking if (a + b) < a is sufficient
        # if using standard addition.
        
        # However, to PROVE it's possible, we look for inputs where this holds.
        # We assume standard addition `+` wraps in BitVec.
        
        is_overflow = z3.BVAddNoOverflow(amount, current_balance, signed=False) # Z3 builtin for "Does it overflow?"
        
        # We want to find a case where it DOES overflow.
        vulnerability = z3.Not(is_overflow)
        
        # Add constraints that inputs are non-zero to be interesting
        return z3.And(vulnerability, amount > 0, current_balance > 0)

class AccessControlViolation(AttackVector):
    name = "Access Control Violation"
    severity = "Critical"
    description = "Sensitive function lacks ownership or role checks."

    def define_predicate(self, state: ContractState) -> z3.ExprRef:
        # Symbolic representation:
        # 1. Function 'withdraw' or 'adminOp' is called
        # 2. msg.sender != owner
        # 3. Call succeeds (no revert constraint generated)
        
        owner = state.declare_storage_var("owner", "address")
        is_privileged_op = z3.Bool('is_privileged_operation')
        call_succeeds = z3.Bool('transaction_succeeds')
        
        vulnerability = z3.And(
            is_privileged_op,
            state.msg_sender != owner,
            call_succeeds
        )
        return vulnerability

# Registry of known vectors
KNOWN_VECTORS = {
    "reentrancy": ReentrancyAttack(),
    "overflow": IntegerOverflowAttack(),
    "access_control": AccessControlViolation()
}
