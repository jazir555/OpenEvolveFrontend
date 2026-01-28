// Lean compiler output
// Module: Mathlib.SetTheory.Ordinal.Arithmetic
// Imports: public import Init public import Mathlib.Algebra.GroupWithZero.Divisibility public import Mathlib.Data.Nat.SuccPred public import Mathlib.Order.SuccPred.InitialSeg public import Mathlib.SetTheory.Ordinal.Basic
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
LEAN_EXPORT lean_object* lp_mathlib_Ordinal_monoid;
LEAN_EXPORT lean_object* lp_mathlib_Ordinal_monoid___lam__0(lean_object*, lean_object*);
lean_object* lp_mathlib_npowBinRecAuto___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_Ordinal_monoidWithZero___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Ordinal_monoid___lam__0___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Ordinal_monoidWithZero;
LEAN_EXPORT lean_object* lp_mathlib_Ordinal_monoid___lam__0(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lean_box(0);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Ordinal_monoid___lam__0___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_Ordinal_monoid___lam__0(x_1, x_2);
lean_dec(x_2);
lean_dec(x_1);
return x_3;
}
}
static lean_object* _init_lp_mathlib_Ordinal_monoid() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; lean_object* x_4; 
x_1 = lean_alloc_closure((void*)(lp_mathlib_Ordinal_monoid___lam__0___boxed), 2, 0);
x_2 = lean_box(0);
lean_inc_ref(x_1);
x_3 = lean_alloc_closure((void*)(lp_mathlib_npowBinRecAuto___boxed), 5, 3);
lean_closure_set(x_3, 0, lean_box(0));
lean_closure_set(x_3, 1, x_1);
lean_closure_set(x_3, 2, x_2);
x_4 = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(x_4, 0, x_1);
lean_ctor_set(x_4, 1, x_2);
lean_ctor_set(x_4, 2, x_3);
return x_4;
}
}
static lean_object* _init_lp_mathlib_Ordinal_monoidWithZero___closed__0() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lean_box(0);
x_2 = lp_mathlib_Ordinal_monoid;
x_3 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_3, 0, x_2);
lean_ctor_set(x_3, 1, x_1);
return x_3;
}
}
static lean_object* _init_lp_mathlib_Ordinal_monoidWithZero() {
_start:
{
lean_object* x_1; 
x_1 = lp_mathlib_Ordinal_monoidWithZero___closed__0;
return x_1;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_GroupWithZero_Divisibility(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Nat_SuccPred(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Order_SuccPred_InitialSeg(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_SetTheory_Ordinal_Basic(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_SetTheory_Ordinal_Arithmetic(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_GroupWithZero_Divisibility(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Nat_SuccPred(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_SuccPred_InitialSeg(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_SetTheory_Ordinal_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_Ordinal_monoid = _init_lp_mathlib_Ordinal_monoid();
lean_mark_persistent(lp_mathlib_Ordinal_monoid);
lp_mathlib_Ordinal_monoidWithZero___closed__0 = _init_lp_mathlib_Ordinal_monoidWithZero___closed__0();
lean_mark_persistent(lp_mathlib_Ordinal_monoidWithZero___closed__0);
lp_mathlib_Ordinal_monoidWithZero = _init_lp_mathlib_Ordinal_monoidWithZero();
lean_mark_persistent(lp_mathlib_Ordinal_monoidWithZero);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
