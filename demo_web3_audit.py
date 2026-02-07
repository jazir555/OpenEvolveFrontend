"""
Demo: Web3 Smart Contract Audit Engine

This script demonstrates the "Lean 4 + Z3" proof system for finding
smart contract violations.
"""

import asyncio
import json
from web3_validator_tool import solve_smart_contract_witness
from automated_proof_engine import create_proof_engine

async def run_demo():
    print("=" * 70)
    print("OpenEvolve: Web3 Smart Contract Audit Engine Demo")
    print("=" * 70)

    # 1. Direct Tool Usage
    print("\n[Step 1] Direct Tool Call: Finding a Reentrancy Witness")
    witness = solve_smart_contract_witness(
        vulnerability_type="reentrancy",
        constraints=["contract balance > 1000", "locked == false"]
    )
    print(f"Success: {witness.get('success')}")
    print(f"Vulnerability: {witness.get('vulnerability')}")
    print(f"Severity: {witness.get('severity')}")
    print(f"Exploit Witness (Z3 Model): {json.dumps(witness.get('witness'), indent=2)}")
    print(f"Remediation: {witness.get('remediation')}")

    # 2. Automated Proof Engine Usage
    print("\n" + "-" * 70)
    print("[Step 2] Automated Proof Engine: Natural Language Audit")
    engine = create_proof_engine()
    
    audit_query = "Prove that a reentrancy attack is possible if the smart contract balance is greater than 0 and the reentrancy lock is disabled."
    
    result = await engine.auto_prove(audit_query, verbose=True)
    
    if result.success:
        print("\n[PROVED]")
        print(f"Theorem: {result.theorem}")
        print(f"Strategy: {result.strategy_used.value}")
        print(f"Evidence:\n{result.final_proof}")
    else:
        print(f"\n[FAILED] {result.error_message}")

if __name__ == "__main__":
    asyncio.run(run_demo())