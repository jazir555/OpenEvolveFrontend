// Lean compiler output
// Module: Mathlib.RingTheory.Polynomial.Chebyshev
// Imports: public import Init public import Mathlib.Algebra.Polynomial.AlgebraMap public import Mathlib.Algebra.Polynomial.Derivative public import Mathlib.Algebra.Polynomial.Degree.Lemmas public import Mathlib.Algebra.Ring.NegOnePow public import Mathlib.Tactic.LinearCombination
#include <lean/lean.h>
#if defined(__clang__)
#pragma clang diagnostic ignored "-Wunused-parameter"
#pragma clang diagnostic ignored "-Wunused-label"
#elif defined(__GNUC__) && !defined(__CLANG__)
#pragma GCC diagnostic ignored "-Wunused-parameter"
#pragma GCC diagnostic ignored "-Wunused-label"
#pragma GCC diagnostic ignored "-Wunused-but-set-variable"
#endif
#ifdef __cplusplus
extern "C" {
#endif
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_RingTheory_Polynomial_Chebyshev_0__Polynomial_Chebyshev_T_match__1_splitter(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_nat_to_int(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_RingTheory_Polynomial_Chebyshev_0__Polynomial_Chebyshev_T_match__1_splitter___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_RingTheory_Polynomial_Chebyshev_0__Polynomial_Chebyshev_T_match__1_splitter___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_nat_abs(lean_object*);
uint8_t lean_nat_dec_eq(lean_object*, lean_object*);
uint8_t lean_int_dec_lt(lean_object*, lean_object*);
lean_object* lean_nat_sub(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_RingTheory_Polynomial_Chebyshev_0__Polynomial_Chebyshev_T_match__1_splitter___redArg___closed__0;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_RingTheory_Polynomial_Chebyshev_0__Polynomial_Chebyshev_T_match__1_splitter___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* _init_lp_mathlib___private_Mathlib_RingTheory_Polynomial_Chebyshev_0__Polynomial_Chebyshev_T_match__1_splitter___redArg___closed__0() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lean_unsigned_to_nat(0u);
x_2 = lean_nat_to_int(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_RingTheory_Polynomial_Chebyshev_0__Polynomial_Chebyshev_T_match__1_splitter___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; lean_object* x_7; uint8_t x_8; 
x_6 = lean_unsigned_to_nat(0u);
x_7 = lp_mathlib___private_Mathlib_RingTheory_Polynomial_Chebyshev_0__Polynomial_Chebyshev_T_match__1_splitter___redArg___closed__0;
x_8 = lean_int_dec_lt(x_1, x_7);
if (x_8 == 0)
{
lean_object* x_9; uint8_t x_10; 
lean_dec(x_5);
x_9 = lean_nat_abs(x_1);
x_10 = lean_nat_dec_eq(x_9, x_6);
if (x_10 == 1)
{
lean_object* x_11; lean_object* x_12; 
lean_dec(x_9);
lean_dec(x_4);
lean_dec(x_3);
x_11 = lean_box(0);
x_12 = lean_apply_1(x_2, x_11);
return x_12;
}
else
{
lean_object* x_13; lean_object* x_14; uint8_t x_15; 
lean_dec(x_2);
x_13 = lean_unsigned_to_nat(1u);
x_14 = lean_nat_sub(x_9, x_13);
lean_dec(x_9);
x_15 = lean_nat_dec_eq(x_14, x_6);
if (x_15 == 1)
{
lean_object* x_16; lean_object* x_17; 
lean_dec(x_14);
lean_dec(x_4);
x_16 = lean_box(0);
x_17 = lean_apply_1(x_3, x_16);
return x_17;
}
else
{
lean_object* x_18; lean_object* x_19; 
lean_dec(x_3);
x_18 = lean_nat_sub(x_14, x_13);
lean_dec(x_14);
x_19 = lean_apply_1(x_4, x_18);
return x_19;
}
}
}
else
{
lean_object* x_20; lean_object* x_21; lean_object* x_22; lean_object* x_23; 
lean_dec(x_4);
lean_dec(x_3);
lean_dec(x_2);
x_20 = lean_nat_abs(x_1);
x_21 = lean_unsigned_to_nat(1u);
x_22 = lean_nat_sub(x_20, x_21);
lean_dec(x_20);
x_23 = lean_apply_1(x_5, x_22);
return x_23;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_RingTheory_Polynomial_Chebyshev_0__Polynomial_Chebyshev_T_match__1_splitter(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_mathlib___private_Mathlib_RingTheory_Polynomial_Chebyshev_0__Polynomial_Chebyshev_T_match__1_splitter___redArg(x_2, x_3, x_4, x_5, x_6);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_RingTheory_Polynomial_Chebyshev_0__Polynomial_Chebyshev_T_match__1_splitter___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_mathlib___private_Mathlib_RingTheory_Polynomial_Chebyshev_0__Polynomial_Chebyshev_T_match__1_splitter(x_1, x_2, x_3, x_4, x_5, x_6);
lean_dec(x_2);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_RingTheory_Polynomial_Chebyshev_0__Polynomial_Chebyshev_T_match__1_splitter___redArg___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib___private_Mathlib_RingTheory_Polynomial_Chebyshev_0__Polynomial_Chebyshev_T_match__1_splitter___redArg(x_1, x_2, x_3, x_4, x_5);
lean_dec(x_1);
return x_6;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Polynomial_AlgebraMap(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Polynomial_Derivative(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Polynomial_Degree_Lemmas(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Ring_NegOnePow(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_LinearCombination(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_RingTheory_Polynomial_Chebyshev(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Polynomial_AlgebraMap(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Polynomial_Derivative(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Polynomial_Degree_Lemmas(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Ring_NegOnePow(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_LinearCombination(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib___private_Mathlib_RingTheory_Polynomial_Chebyshev_0__Polynomial_Chebyshev_T_match__1_splitter___redArg___closed__0 = _init_lp_mathlib___private_Mathlib_RingTheory_Polynomial_Chebyshev_0__Polynomial_Chebyshev_T_match__1_splitter___redArg___closed__0();
lean_mark_persistent(lp_mathlib___private_Mathlib_RingTheory_Polynomial_Chebyshev_0__Polynomial_Chebyshev_T_match__1_splitter___redArg___closed__0);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
