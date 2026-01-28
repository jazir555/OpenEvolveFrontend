// Lean compiler output
// Module: Mathlib.RingTheory.Polynomial.Content
// Imports: public import Init public import Mathlib.Algebra.GCDMonoid.Finset public import Mathlib.Algebra.Polynomial.CancelLeads public import Mathlib.Algebra.Polynomial.EraseLead public import Mathlib.Algebra.Polynomial.FieldDivision
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
lean_object* lp_mathlib_Finset_gcd___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_CommSemiring_toCommMonoidWithZero___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Polynomial_content(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Polynomial_content___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Polynomial_coeff___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Polynomial_content___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; 
x_4 = lean_ctor_get(x_1, 0);
lean_inc_ref(x_4);
lean_dec_ref(x_1);
lean_inc_ref(x_4);
x_5 = lp_mathlib_CommSemiring_toCommMonoidWithZero___redArg(x_4);
x_6 = lean_ctor_get(x_3, 0);
lean_inc(x_6);
x_7 = lean_alloc_closure((void*)(lp_mathlib_Polynomial_coeff___boxed), 4, 3);
lean_closure_set(x_7, 0, lean_box(0));
lean_closure_set(x_7, 1, x_4);
lean_closure_set(x_7, 2, x_3);
x_8 = lp_mathlib_Finset_gcd___redArg(x_5, x_2, x_6, x_7);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Polynomial_content(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_Polynomial_content___redArg(x_2, x_4, x_5);
return x_6;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_GCDMonoid_Finset(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Polynomial_CancelLeads(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Polynomial_EraseLead(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Polynomial_FieldDivision(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_RingTheory_Polynomial_Content(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_GCDMonoid_Finset(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Polynomial_CancelLeads(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Polynomial_EraseLead(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Polynomial_FieldDivision(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
