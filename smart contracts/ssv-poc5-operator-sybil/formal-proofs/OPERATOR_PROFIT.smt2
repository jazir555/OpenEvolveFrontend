; Operator Sybil Profitability Proof (POC 5)
; Proves that Profit > Investment is reachable (Infinite Money Glitch)

(declare-const investment Int)
(declare-const profit Int)
(declare-const time Int)
(declare-const sybil_count Int)
(declare-const fee Int)

; Constants
(assert (= sybil_count 50))
(assert (= fee 1))
(assert (= investment 250)) ; 50 sybils * 5 SSV

; Profit Calculation
; Profit = (Sybil_Count * Fee * Time) - Investment
(assert (= profit (- (* sybil_count (* fee time)) investment)))

; Time passes beyond bankruptcy point
(assert (> time 10))

; Goal: Profit is positive and grows
(assert (> profit 0))
(assert (> profit investment)) ; > 100% ROI

(check-sat)
(get-model)
