import z3
import os

PROOF_FILES = [
    "ssv-insolvency-poc/formal-proofs/SSV_INSOLVENCY_PROOF.smt2",
    "ssv-poc2-multi-cluster/formal-proofs/MULTI_CLUSTER_INSOLVENCY_PROOF.smt2",
    "ssv-poc3-liquidation-griefing/formal-proofs/LIQUIDATION_GRIEFING_PROOF.smt2",
    "ssv-poc4-dao-sybil/formal-proofs/DAO_INSOLVENCY.smt2",
    "ssv-poc5-operator-sybil/formal-proofs/OPERATOR_PROFIT.smt2"
]

def verify_proof(relative_path):
    # Resolve path relative to this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, relative_path)
    
    print(f"\n[VERIFYING] {relative_path}...")
    try:
        formulas = z3.parse_smt2_file(file_path)
        s = z3.Solver()
        s.add(formulas)
        result = s.check()
        
        if result == z3.sat:
            print(">>> RESULT: SAT (Vulnerability Reachable)")
            print(">>> Model Witness:")
            print(s.model())
            return True
        elif result == z3.unsat:
            print(">>> RESULT: UNSAT (Model Secure - Exploit Failed)")
            return False
        else:
            print(">>> RESULT: UNKNOWN")
            return False
    except Exception as e:
        print(f"!!! ERROR: {e}")
        return False

if __name__ == "__main__":
    print("=================================================================")
    print("SSV INSOLVENCY: Z3 FORMAL VERIFICATION SUITE")
    print("=================================================================")
    
    success_count = 0
    for proof in PROOF_FILES:
        if verify_proof(proof):
            success_count += 1
            
    print("\n=================================================================")
    print(f"SUMMARY: {success_count}/{len(PROOF_FILES)} Proofs Confirmed Vulnerability.")
    print("=================================================================")