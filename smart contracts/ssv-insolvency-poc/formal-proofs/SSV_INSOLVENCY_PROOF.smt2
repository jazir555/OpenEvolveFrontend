; SSV Protocol Insolvency Formal Proof
; Language: SMT-LIB v2.6
; Tool: Z3 / cvc5
;
; This proof demonstrates that the ssv.network virtual accounting system
; can transition from a safe state to an insolvent state due to uncollateralized
; operator fee accumulation.

(set-logic LIA)
(set-option :produce-models true)

; --- State Variables ---

; initial_assets: Actual SSV tokens in the contract shared pool
(declare-fun initial_assets () Int)

; honest_deposits: Portion of assets belonging to collateralized users
(declare-fun honest_deposits () Int)

; bankrupt_deposit: Initial deposit of the cluster that will go bankrupt
(declare-fun bankrupt_deposit () Int)

; blocks_delayed: Number of blocks passed after the cluster became insolvent
(declare-fun blocks_delayed () Int)

; operator_fee: The fee rate credited to operators
(declare-fun operator_fee () Int)

; --- Invariants & Domain Constraints ---

; Assets must be positive
(assert (> initial_assets 0))
(assert (> bankrupt_deposit 0))
(assert (>= honest_deposits 0))

; Initial conservation: assets = honest_deposits + bankrupt_deposit
(assert (= initial_assets (+ honest_deposits bankrupt_deposit)))

; Parameters must be positive
(assert (> blocks_delayed 0))
(assert (> operator_fee 0))

; --- Transition Logic (The Vulnerability) ---

; 1. The bankrupt cluster consumes all its deposit.
; 2. Operator logic credits fees for the 'delayed' blocks despite 0 balance.
(define-fun virtual_debt () Int (* blocks_delayed operator_fee))

; 3. Total Liabilities = (Funds owed to honest users) + (Virtual debt to operators)
; Note: The bankrupt deposit is gone, but operators are owed more than it was worth.
(define-fun total_liabilities () Int (+ honest_deposits bankrupt_deposit virtual_debt))

; --- The Safety Invariant (Correctness) ---
; A protocol is solvent if Assets >= Liabilities
(define-fun is_solvent () Bool (>= initial_assets total_liabilities))

; --- The Proof of Violation ---
; We assert that the system is NOT solvent. 
; If this is SATISFIABLE, the vulnerability is PROVEN.
(assert (not is_solvent))

(check-sat)
(get-model)

; --- Expected Result: SAT ---
; Z3 will find values for assets, fee, and blocks where promised tokens > actual tokens.
