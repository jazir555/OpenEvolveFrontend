import z3
import sys

def run_smt_proof(filepath):
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