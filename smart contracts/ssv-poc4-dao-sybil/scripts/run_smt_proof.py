import z3
import sys
import os

def run_smt_proof(filepath):
    if not os.path.isabs(filepath):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        filepath = os.path.normpath(os.path.join(script_dir, filepath))

    print(f"Executing SMT-LIB Proof: {filepath}")
    s = z3.Solver()
    try:
        s.from_file(filepath)
        result = s.check()
        print(f"Result: {result}")
        if result == z3.sat:
            print("[VULNERABILITY PROVEN] Exploit state is satisfiable.")
            print("Satisfying Model Witness:")
            print(str(s.model()))
        else:
            print("[UNPROVED] State is not satisfiable.")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    run_smt_proof("../formal-proofs/DAO_INSOLVENCY.smt2")
