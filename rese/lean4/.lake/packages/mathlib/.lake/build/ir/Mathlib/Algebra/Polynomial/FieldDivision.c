// Lean compiler output
// Module: Mathlib.Algebra.Polynomial.FieldDivision
// Imports: public import Init public import Mathlib.Algebra.Order.Group.Finset public import Mathlib.Algebra.Polynomial.Derivative public import Mathlib.Algebra.Polynomial.Eval.SMul public import Mathlib.Algebra.Polynomial.Roots public import Mathlib.RingTheory.EuclideanDomain public import Mathlib.RingTheory.UniqueFactorizationDomain.NormalizedFactors
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
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Order_Group_Finset(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Polynomial_Derivative(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Polynomial_Eval_SMul(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Polynomial_Roots(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_RingTheory_EuclideanDomain(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_RingTheory_UniqueFactorizationDomain_NormalizedFactors(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Algebra_Polynomial_FieldDivision(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Order_Group_Finset(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Polynomial_Derivative(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Polynomial_Eval_SMul(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Polynomial_Roots(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_RingTheory_EuclideanDomain(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_RingTheory_UniqueFactorizationDomain_NormalizedFactors(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
