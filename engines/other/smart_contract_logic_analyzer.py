"""
Smart Contract Logic Analyzer

This module provides the "Translation Layer" between high-level Smart Contract concepts
(Solidity-like) and low-level Z3 SMT constraints. It allows for the symbolic execution
and verification of smart contract logic.

It defines symbolic representations for:
- EVM State (Balances, Storage)
- Transactions
- Common vulnerability predicates

Author: OpenEvolve
"""

import logging
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
import z3

logger = logging.getLogger(__name__)

@dataclass
class Symbol:
    """A symbolic variable in the contract state."""
    name: str
    sort: z3.SortRef
    ref: z3.ExprRef

class ContractState:
    """
    Represents the symbolic state of a smart contract at a given point in time.
    """
    def __init__(self, solver: z3.Solver):
        self.solver = solver
        self.variables: Dict[str, Symbol] = {}
        self.constraints: List[z3.ExprRef] = []
        
        # Standard EVM State Variables
        self._init_global_vars()

    def _init_global_vars(self):
        """Initialize standard EVM global variables."""
        # Address type as BitVec(160)
        self.msg_sender = z3.BitVec('msg.sender', 160)
        self.contract_address = z3.BitVec('address(this)', 160)
        
        # Balances (Map: Address -> Int)
        # Using Array for mapping representation in Z3
        # In a real EVM, balances are uint256, but Int is often easier for proofs unless overflow is the target
        self.balances = z3.Array('balances', z3.BitVecSort(160), z3.IntSort())
        
        # Block variables
        self.block_timestamp = z3.Int('block.timestamp')
        self.block_number = z3.Int('block.number')
        
        # Add basic constraints (non-negative balances, etc.)
        self.solver.add(self.balances[self.msg_sender] >= 0)
        self.solver.add(self.balances[self.contract_address] >= 0)
        self.solver.add(self.block_timestamp > 0)

    def declare_storage_var(self, name: str, sort_type: str = "int") -> z3.ExprRef:
        """
        Declare a symbolic storage variable for the contract.
        
        Args:
            name: Variable name
            sort_type: "int", "bool", "address", or "uint256"
            
        Returns:
            Z3 expression reference
        """
        if name in self.variables:
            return self.variables[name].ref
            
        if sort_type == "int":
            sort = z3.IntSort()
            ref = z3.Int(name)
        elif sort_type == "bool":
            sort = z3.BoolSort()
            ref = z3.Bool(name)
        elif sort_type == "address":
            sort = z3.BitVecSort(160)
            ref = z3.BitVec(name, 160)
        elif sort_type == "uint256":
            sort = z3.BitVecSort(256)
            ref = z3.BitVec(name, 256)
        else:
            raise ValueError(f"Unsupported sort type: {sort_type}")
            
        self.variables[name] = Symbol(name, sort, ref)
        return ref

    def get_var(self, name: str) -> z3.ExprRef:
        """Get a variable by name."""
        if name in self.variables:
            return self.variables[name].ref
        # Fallback for dynamic lookup if not explicitly declared but standard
        if name == "msg.sender": return self.msg_sender
        if name == "address(this)": return self.contract_address
        if name == "block.timestamp": return self.block_timestamp
        raise ValueError(f"Variable {name} not found")

    def add_constraint(self, expr: z3.ExprRef, description: str = ""):
        """Add a constraint to the solver."""
        self.constraints.append(expr)
        self.solver.add(expr)
        if description:
            logger.debug(f"Added constraint: {description}")

    def transfer(self, from_addr: z3.ExprRef, to_addr: z3.ExprRef, amount: z3.ExprRef) -> z3.BoolRef:
        """
        Create a state transition representing a transfer.
        Returns the condition under which the transfer is valid (sufficient balance).
        Note: This acts as a 'require' in Solidity.
        """
        # Precondition: Sufficient balance
        sufficient_balance = (self.balances[from_addr] >= amount)
        
        # State update (functional style - creates new array state)
        new_balances = z3.Store(self.balances, from_addr, self.balances[from_addr] - amount)
        new_balances = z3.Store(new_balances, to_addr, self.balances[to_addr] + amount)
        
        # In a real transition system, we'd update self.balances to new_balances for the next step.
        # For simple verification, we might just assert the relation.
        # Here we return the validity condition.
        return sufficient_balance

class VulnerabilityScanner:
    """
    Base class for scanning vulnerabilities using the ContractState.
    """
    def __init__(self):
        self.solver = z3.Solver()
        self.state = ContractState(self.solver)

    def check_predicate(self, predicate: z3.ExprRef) -> Tuple[bool, Optional[z3.ModelRef]]:
        """
        Check if a predicate can be satisfied (i.e., if an exploit is possible).
        
        Args:
            predicate: The condition representing the vulnerability state.
            
        Returns:
            (is_satisfiable, model)
        """
        # We want to find IF there exists a state where predicate is True.
        self.solver.push()
        self.solver.add(predicate)
        
        result = self.solver.check()
        
        if result == z3.sat:
            model = self.solver.model()
            self.solver.pop()
            return True, model
        else:
            self.solver.pop()
            return False, None

