// Lean compiler output
// Module: Mathlib.RingTheory.MvPolynomial.Localization
// Imports: public import Init public import Mathlib.Algebra.Module.LocalizedModule.IsLocalization public import Mathlib.Algebra.MvPolynomial.CommRing public import Mathlib.RingTheory.Ideal.Quotient.Operations public import Mathlib.RingTheory.Localization.Away.Basic public import Mathlib.RingTheory.Localization.BaseChange public import Mathlib.RingTheory.TensorProduct.MvPolynomial
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
lean_object* initialize_mathlib_Mathlib_Algebra_Module_LocalizedModule_IsLocalization(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_MvPolynomial_CommRing(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_RingTheory_Ideal_Quotient_Operations(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_RingTheory_Localization_Away_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_RingTheory_Localization_BaseChange(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_RingTheory_TensorProduct_MvPolynomial(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_RingTheory_MvPolynomial_Localization(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Module_LocalizedModule_IsLocalization(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_MvPolynomial_CommRing(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_RingTheory_Ideal_Quotient_Operations(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_RingTheory_Localization_Away_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_RingTheory_Localization_BaseChange(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_RingTheory_TensorProduct_MvPolynomial(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
