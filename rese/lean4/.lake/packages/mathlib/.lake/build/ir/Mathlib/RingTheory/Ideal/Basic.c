// Lean compiler output
// Module: Mathlib.RingTheory.Ideal.Basic
// Imports: public import Init public import Mathlib.Algebra.Field.IsField public import Mathlib.Data.Fin.VecNotation public import Mathlib.Data.Nat.Choose.Sum public import Mathlib.LinearAlgebra.Finsupp.LinearCombination public import Mathlib.RingTheory.Ideal.Maximal public import Mathlib.Tactic.FinCases
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
LEAN_EXPORT lean_object* lp_mathlib_Ideal_equivFinTwo___redArg___lam__0(lean_object*, lean_object*);
static lean_object* lp_mathlib_Ideal_equivFinTwo___redArg___lam__0___closed__1;
static lean_object* lp_mathlib_Ideal_equivFinTwo___redArg___closed__0;
lean_object* lp_mathlib_Matrix_vecCons___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Matrix_vecEmpty___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Ideal_pi___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Ideal_equivFinTwo___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Ideal_pi(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_nat_mod(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Ideal_equivFinTwo(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Ideal_equivFinTwo___redArg(lean_object*);
static lean_object* lp_mathlib_Ideal_equivFinTwo___redArg___lam__0___closed__0;
static lean_object* lp_mathlib_Ideal_equivFinTwo___redArg___closed__1;
static lean_object* lp_mathlib_Ideal_equivFinTwo___redArg___closed__2;
LEAN_EXPORT lean_object* lp_mathlib_Ideal_pi(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lean_box(0);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Ideal_pi___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_Ideal_pi(x_1, x_2, x_3, x_4);
lean_dec_ref(x_4);
lean_dec_ref(x_3);
return x_5;
}
}
static lean_object* _init_lp_mathlib_Ideal_equivFinTwo___redArg___lam__0___closed__0() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lean_unsigned_to_nat(2u);
x_2 = lean_unsigned_to_nat(1u);
x_3 = lean_nat_mod(x_2, x_1);
return x_3;
}
}
static lean_object* _init_lp_mathlib_Ideal_equivFinTwo___redArg___lam__0___closed__1() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lean_unsigned_to_nat(2u);
x_2 = lean_unsigned_to_nat(0u);
x_3 = lean_nat_mod(x_2, x_1);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Ideal_equivFinTwo___redArg___lam__0(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; uint8_t x_5; 
x_3 = lean_box(0);
x_4 = lean_apply_2(x_1, x_2, x_3);
x_5 = lean_unbox(x_4);
if (x_5 == 0)
{
lean_object* x_6; 
x_6 = lp_mathlib_Ideal_equivFinTwo___redArg___lam__0___closed__0;
return x_6;
}
else
{
lean_object* x_7; 
x_7 = lp_mathlib_Ideal_equivFinTwo___redArg___lam__0___closed__1;
return x_7;
}
}
}
static lean_object* _init_lp_mathlib_Ideal_equivFinTwo___redArg___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_closure((void*)(lp_mathlib_Matrix_vecEmpty___boxed), 2, 1);
lean_closure_set(x_1, 0, lean_box(0));
return x_1;
}
}
static lean_object* _init_lp_mathlib_Ideal_equivFinTwo___redArg___closed__1() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; lean_object* x_4; 
x_1 = lp_mathlib_Ideal_equivFinTwo___redArg___closed__0;
x_2 = lean_box(0);
x_3 = lean_unsigned_to_nat(0u);
x_4 = lean_alloc_closure((void*)(lp_mathlib_Matrix_vecCons___boxed), 5, 4);
lean_closure_set(x_4, 0, lean_box(0));
lean_closure_set(x_4, 1, x_3);
lean_closure_set(x_4, 2, x_2);
lean_closure_set(x_4, 3, x_1);
return x_4;
}
}
static lean_object* _init_lp_mathlib_Ideal_equivFinTwo___redArg___closed__2() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; lean_object* x_4; 
x_1 = lp_mathlib_Ideal_equivFinTwo___redArg___closed__1;
x_2 = lean_box(0);
x_3 = lean_unsigned_to_nat(1u);
x_4 = lean_alloc_closure((void*)(lp_mathlib_Matrix_vecCons___boxed), 5, 4);
lean_closure_set(x_4, 0, lean_box(0));
lean_closure_set(x_4, 1, x_3);
lean_closure_set(x_4, 2, x_2);
lean_closure_set(x_4, 3, x_1);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Ideal_equivFinTwo___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; 
x_2 = lean_alloc_closure((void*)(lp_mathlib_Ideal_equivFinTwo___redArg___lam__0), 2, 1);
lean_closure_set(x_2, 0, x_1);
x_3 = lp_mathlib_Ideal_equivFinTwo___redArg___closed__2;
x_4 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_4, 0, x_2);
lean_ctor_set(x_4, 1, x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Ideal_equivFinTwo(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_Ideal_equivFinTwo___redArg(x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Ideal_equivFinTwo___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_Ideal_equivFinTwo(x_1, x_2, x_3);
lean_dec_ref(x_2);
return x_4;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Field_IsField(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Fin_VecNotation(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Nat_Choose_Sum(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_LinearAlgebra_Finsupp_LinearCombination(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_RingTheory_Ideal_Maximal(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_FinCases(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_RingTheory_Ideal_Basic(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Field_IsField(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Fin_VecNotation(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Nat_Choose_Sum(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_LinearAlgebra_Finsupp_LinearCombination(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_RingTheory_Ideal_Maximal(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_FinCases(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_Ideal_equivFinTwo___redArg___lam__0___closed__0 = _init_lp_mathlib_Ideal_equivFinTwo___redArg___lam__0___closed__0();
lean_mark_persistent(lp_mathlib_Ideal_equivFinTwo___redArg___lam__0___closed__0);
lp_mathlib_Ideal_equivFinTwo___redArg___lam__0___closed__1 = _init_lp_mathlib_Ideal_equivFinTwo___redArg___lam__0___closed__1();
lean_mark_persistent(lp_mathlib_Ideal_equivFinTwo___redArg___lam__0___closed__1);
lp_mathlib_Ideal_equivFinTwo___redArg___closed__0 = _init_lp_mathlib_Ideal_equivFinTwo___redArg___closed__0();
lean_mark_persistent(lp_mathlib_Ideal_equivFinTwo___redArg___closed__0);
lp_mathlib_Ideal_equivFinTwo___redArg___closed__1 = _init_lp_mathlib_Ideal_equivFinTwo___redArg___closed__1();
lean_mark_persistent(lp_mathlib_Ideal_equivFinTwo___redArg___closed__1);
lp_mathlib_Ideal_equivFinTwo___redArg___closed__2 = _init_lp_mathlib_Ideal_equivFinTwo___redArg___closed__2();
lean_mark_persistent(lp_mathlib_Ideal_equivFinTwo___redArg___closed__2);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
