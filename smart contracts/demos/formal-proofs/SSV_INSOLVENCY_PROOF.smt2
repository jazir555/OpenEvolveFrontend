; SSV Protocol Insolvency Formal Proof (Finalized)
; Language: SMT-LIB v2.6
;
; This proof demonstrates that the ssv.network virtual accounting system
; transition leads to protocol insolvency using the finalized PoC parameters.

(set-logic LIA)
(set-option :produce-models true)

; --- State Variables ---
(declare-fun honest_deposit () Int)   ; User A
(declare-fun bankrupt_deposit () Int) ; User B
(declare-fun blocks_passed () Int)    ; Time since both started
(declare-fun operator_fee () Int)     ; Fee per block

; --- Protocol Logic Functions ---

; Assets: Total tokens held in the shared contract pool
(define-fun total_assets () Int (+ honest_deposit bankrupt_deposit))

; Liabilities: Sum of all virtual promises
; 1. Honest User is owed their full deposit (as they are collateralized)
; 2. Operator is owed fees for the entire duration (unconditional credit)
(define-fun operator_earnings () Int (* blocks_passed operator_fee))
(define-fun total_liabilities () Int (+ honest_deposit operator_earnings))

; --- PoC Parameters (Matching Foundry Trace) ---
(assert (= honest_deposit 1000))
(assert (= bankrupt_deposit 10))
(assert (= blocks_passed 10))
(assert (= operator_fee 5))

; --- The Safety Invariant Violation ---
; System is insolvent if Liabilities > Assets
(define-fun is_insolvent () Bool (> total_liabilities total_assets))

; --- The Proof ---
; We prove that given the protocol rules, insolvency is TRUE for these parameters.
(assert is_insolvent)

(check-sat)
(get-model)

; --- Result Interpretation ---
; Result: SAT
; This confirms that with 1010 tokens in the pool, 
; the protocol promises 1000 (User A) + 50 (Operator) = 1050 tokens.
; Deficit = 40 tokens.