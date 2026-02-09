import z3
import sys
import os

def run_smt_proof(filepath):
    # Resolve path relative to this script if it's a relative path
    if not os.path.isabs(filepath):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        # If the filepath already starts with "smart contracts/", strip it to avoid double-nesting
        # when running from the script's directory
        if filepath.startswith("smart contracts/"):
            filepath = filepath[len("smart contracts/"):]
        filepath = os.path.join(script_dir, filepath)

    print("Executing SMT-LIB Proof: " + str(filepath))
    s = z3.Solver()
    try:
        s.from_file(filepath)
        result = s.check()
        print("Result: " + str(result))
        if result == z3.sat:
            print("[VULNERABILITY PROVEN] Exploit state is satisfiable.")
            print("Satisfying Model:")
            print(str(s.model()))
        else:
            print("[UNPROVED] State is not satisfiable.")
    except Exception as e:
        print("Error: " + str(e))

if __name__ == "__main__":
    run_smt_proof("smart contracts/SSV_INSOLVENCY_PROOF.smt2")