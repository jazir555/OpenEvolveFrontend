; SSV Multi-Cluster Insolvency Formal Proof
; Language: SMT-LIB v2.6
;
; This proof demonstrates that multiple bankrupt clusters compound
; protocol insolvency using the multi-cluster attack scenario.

(set-logic LIA)
(set-option :produce-models true)

; --- State Variables ---
(declare-fun large_deposit () Int)     ; 10000
(declare-fun small_1_deposit () Int)   ; 100
(declare-fun small_2_deposit () Int)   ; 50
(declare-fun small_3_deposit () Int)   ; 25
(declare-fun blocks_passed () Int)     ; 150
(declare-fun op_fee () Int)            ; 1

; --- Protocol Logic ---

; Assets: Total tokens held in the shared contract pool
(define-fun total_assets () Int (+ large_deposit small_1_deposit small_2_deposit small_3_deposit))

; Virtual debt from each bankrupt cluster:
; Small 1: Bankrupt at block 100, debt for 50 blocks
(define-fun virtual_debt_1 () Int (* 50 op_fee))

; Small 2: Bankrupt at block 50, debt for 100 blocks
(define-fun virtual_debt_2 () Int (* 100 op_fee))

; Small 3: Bankrupt at block 25, debt for 125 blocks
(define-fun virtual_debt_3 () Int (* 125 op_fee))

; Total virtual debt from operators
(define-fun total_virtual_debt () Int (+ virtual_debt_1 virtual_debt_2 virtual_debt_3))

; Liabilities: Large user entitlement + virtual debt
(define-fun total_liabilities () Int (+ large_deposit total_virtual_debt))

; --- PoC Parameters ---
(assert (= large_deposit 10000))
(assert (= small_1_deposit 100))
(assert (= small_2_deposit 50))
(assert (= small_3_deposit 25))
(assert (= blocks_passed 150))
(assert (= op_fee 1))

; --- The Safety Invariant Violation ---
; System is insolvent if Liabilities > Assets
(define-fun is_insolvent () Bool (> total_liabilities total_assets))

; --- The Proof ---
(assert is_insolvent)

(check-sat)
(get-model)

; --- Result Interpretation ---
; Result: SAT
; This confirms that with 10175 tokens in the pool,
; the protocol promises 10000 (Large User) + 275 (Virtual Debt) = 10275 tokens.
; Deficit = 100 tokens. Multi-cluster insolvency is proven.
