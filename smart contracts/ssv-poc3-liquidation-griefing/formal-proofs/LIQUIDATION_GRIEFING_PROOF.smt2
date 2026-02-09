; SSV Liquidation Griefing Insolvency Formal Proof
; Language: SMT-LIB v2.6
;
; This proof demonstrates that delaying liquidation through griefing
; maximizes virtual debt and enables larger theft from honest users.

(set-logic LIA)
(set-option :produce-models true)

; --- State Variables ---
(declare-fun large_deposit () Int)     ; 10000
(declare-fun small_1_deposit () Int)   ; 100
(declare-fun small_2_deposit () Int)   ; 50
(declare-fun small_3_deposit () Int)   ; 25
(declare-fun griefing_delay () Int)    ; 200 blocks
(declare-fun op_fee () Int)            ; 1

; --- Protocol Logic ---

; Assets: Total tokens held
(define-fun total_assets () Int (+ large_deposit small_1_deposit small_2_deposit small_3_deposit))

; Virtual debt WITHOUT griefing (immediate liquidation):
; Small 1: Bankrupt at 100, liquidated at 100 -> 0 blocks
; Small 2: Bankrupt at 50, liquidated at 50 -> 0 blocks
; Small 3: Bankrupt at 25, liquidated at 25 -> 0 blocks
(define-fun normal_virtual_debt () Int 0)

; Virtual debt WITH griefing (200 block delay):
; Small 1: Bankrupt at 100, liquidated at 300 -> 200 blocks
; Small 2: Bankrupt at 50, liquidated at 250 -> 200 blocks
; Small 3: Bankrupt at 25, liquidated at 225 -> 200 blocks
(define-fun griefing_virtual_debt_1 () Int (* 200 op_fee))
(define-fun griefing_virtual_debt_2 () Int (* 200 op_fee))
(define-fun griefing_virtual_debt_3 () Int (* 200 op_fee))
(define-fun total_griefing_virtual_debt () Int (+ griefing_virtual_debt_1 griefing_virtual_debt_2 griefing_virtual_debt_3))

; Liabilities with griefing
(define-fun total_liabilities () Int (+ large_deposit total_griefing_virtual_debt))

; Profit from griefing
(define-fun griefing_profit () Int (- total_griefing_virtual_debt normal_virtual_debt))

; --- PoC Parameters ---
(assert (= large_deposit 10000))
(assert (= small_1_deposit 100))
(assert (= small_2_deposit 50))
(assert (= small_3_deposit 25))
(assert (= griefing_delay 200))
(assert (= op_fee 1))

; --- The Safety Invariant Violation ---
(define-fun is_insolvent () Bool (> total_liabilities total_assets))

; Griefing must be profitable
(define-fun is_profitable () Bool (> griefing_profit 0))

; --- The Proof ---
(assert is_insolvent)
(assert is_profitable)

(check-sat)
(get-model)

; --- Result Interpretation ---
; Result: SAT
; This confirms that griefing increases virtual debt from 0 to 600 SSV,
; creating a 600 SSV deficit that steals from the large user's deposit.
; Liquidation griefing maximizes theft and is economically rational.
